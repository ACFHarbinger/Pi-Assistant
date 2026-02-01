import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-shell";
import { useAgentStore } from "../stores/agentStore";
import { useClientAI } from "../hooks/useClientAI";

export default function Settings({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const {
    loadModel,
    unloadModel,
    fetchModels: refreshGlobalModels,
  } = useAgentStore();
  const [activeTab, setActiveTab] = useState<
    | "mcp"
    | "tools"
    | "models"
    | "auth"
    | "channels"
    | "agents"
    | "marketplace"
    | "reset"
  >("mcp");
  const [mcpConfig, setMcpConfig] = useState<any>({});
  const [toolsConfig, setToolsConfig] = useState<any>({});
  const [localModels, setLocalModels] = useState<any[]>([]);
  const [downloading, setDownloading] = useState<Record<string, boolean>>({});
  const [searchQuery, setSearchQuery] = useState("");

  // Forms
  const [newServerName, setNewServerName] = useState("");
  const [newServerCmd, setNewServerCmd] = useState("");
  const [newServerArgs, setNewServerArgs] = useState("");

  const [newModelId, setNewModelId] = useState("");
  const [localBackend, setLocalBackend] = useState<
    "transformers" | "llama.cpp" | "auto"
  >("auto");

  // Reset Options
  const [resetOptions, setResetOptions] = useState({
    memory: true,
    mcp_config: false,
    tools_config: false,
    models_config: false,
    personality: true,
  });
  const [isResetting, setIsResetting] = useState(false);
  const [telegramConfig, setTelegramConfig] = useState<any>({
    token: "",
    enabled: false,
    allowed_users: [],
  });
  const [telegramLoading, setTelegramLoading] = useState(false);
  const [discordConfig, setDiscordConfig] = useState<any>({
    token: "",
    enabled: false,
  });
  const [discordLoading, setDiscordLoading] = useState(false);

  // Client AI Test
  const { isLoaded: isClientAILoaded, predict: predictClientAI } =
    useClientAI();
  const [clientAIPrediction, setClientAIPrediction] = useState<string | null>(
    null,
  );

  const handleTestClientAI = () => {
    if (isClientAILoaded) {
      const result = predictClientAI("Hello Client AI");
      console.log("Prediction result:", result);
      setClientAIPrediction(JSON.stringify(result));
    }
  };

  useEffect(() => {
    if (isOpen) {
      refreshConfig();
    }
  }, [isOpen]);

  async function refreshConfig() {
    try {
      const mcp = await invoke("get_mcp_config");
      setMcpConfig(mcp);
      const tools = await invoke("get_tools_config");
      setToolsConfig(tools);
      const local = (await invoke("list_local_models")) as any[];
      setLocalModels(local);
      const tg = await invoke("get_telegram_config");
      setTelegramConfig(tg);
      const dc = await invoke("get_discord_config");
      setDiscordConfig(dc);
      // Sync global model list for ChatInterface
      await refreshGlobalModels();
    } catch (e) {
      console.error("Failed to load config:", e);
    }
  }

  async function handleAddServer() {
    try {
      const args = newServerArgs.split(" ").filter((s) => s.trim().length > 0);
      await invoke("save_mcp_server", {
        name: newServerName,
        config: { command: newServerCmd, args, env: {} },
      });
      setNewServerName("");
      setNewServerCmd("");
      setNewServerArgs("");
      refreshConfig();
    } catch (e) {
      console.error(e);
      alert("Failed to save server");
    }
  }

  async function handleRemoveServer(name: string) {
    if (!confirm(`Remove server ${name}?`)) return;
    await invoke("remove_mcp_server", { name });
    refreshConfig();
  }

  async function handleToggleTool(name: string, current: boolean) {
    await invoke("toggle_tool", { name, enabled: !current });
    refreshConfig();
  }

  async function handleAddModel() {
    await invoke("save_model", {
      model: { id: newModelId, description: "User added" },
    });
    setNewModelId("");
    refreshConfig();
  }

  async function handleLoadModel(id: string) {
    try {
      await loadModel(id, localBackend === "auto" ? null : localBackend);
      await refreshConfig();
      alert("Model loaded!");
    } catch (e) {
      console.error(e);
      alert("Failed to load model: " + e);
    }
  }

  async function handleUnloadModel(id: string) {
    try {
      await unloadModel(id);
      await refreshConfig();
      alert("Model unloaded!");
    } catch (e) {
      console.error(e);
      alert("Failed to unload model: " + e);
    }
  }

  async function handleDownloadModel(id: string) {
    setDownloading((prev) => ({ ...prev, [id]: true }));
    try {
      await invoke("download_model", { modelId: id });
      refreshConfig();
      alert("Download complete!");
    } catch (e) {
      console.error(e);
      alert("Download failed: " + e);
    } finally {
      setDownloading((prev) => ({ ...prev, [id]: false }));
    }
  }

  // Auth connection state
  const [claudeConnected, setClaudeConnected] = useState(false);
  const [claudeLoading, setClaudeLoading] = useState(false);
  const [antigravityConnected, setAntigravityConnected] = useState(false);
  const [antigravityLoading, setAntigravityLoading] = useState(false);
  const [geminiConnected, setGeminiConnected] = useState(false);
  const [geminiLoading, setGeminiLoading] = useState(false);

  useEffect(() => {
    if (isOpen && activeTab === "auth") {
      invoke("check_claude_auth").then((connected: any) =>
        setClaudeConnected(connected),
      );
      invoke("check_provider_auth", { provider: "antigravity" }).then(
        (connected: any) => setAntigravityConnected(connected),
      );
      invoke("check_provider_auth", { provider: "gemini" }).then(
        (connected: any) => setGeminiConnected(connected),
      );
    }
  }, [isOpen, activeTab]);

  async function handleClaudeLogin() {
    setClaudeLoading(true);
    try {
      await invoke("start_claude_oauth");
      setClaudeConnected(true);
      alert("Claude Pro/Max connected successfully!");
    } catch (e) {
      console.error(e);
      alert(`Claude login failed: ${e}`);
    } finally {
      setClaudeLoading(false);
    }
  }

  async function handleClaudeDisconnect() {
    try {
      await invoke("disconnect_claude_auth");
      setClaudeConnected(false);
    } catch (e) {
      console.error(e);
      alert(`Disconnect failed: ${e}`);
    }
  }

  async function handleLogin(provider: string) {
    const setLoading =
      provider === "antigravity" ? setAntigravityLoading : setGeminiLoading;
    const setConnected =
      provider === "antigravity" ? setAntigravityConnected : setGeminiConnected;
    setLoading(true);
    try {
      const code = await invoke("start_oauth", { provider, clientId: "" });
      await invoke("exchange_oauth_code", {
        provider,
        code,
        clientId: "",
        clientSecret: "",
        redirectUri: "",
      });
      setConnected(true);
      alert(`Login to ${provider} successful!`);
    } catch (e) {
      console.error(e);
      alert(`Login failed: ${e}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleProviderDisconnect(provider: string) {
    const setConnected =
      provider === "antigravity" ? setAntigravityConnected : setGeminiConnected;
    try {
      await invoke("disconnect_provider_auth", { provider });
      setConnected(false);
    } catch (e) {
      console.error(e);
      alert(`Disconnect failed: ${e}`);
    }
  }

  async function handleReset() {
    const count = Object.values(resetOptions).filter(Boolean).length;
    if (count === 0) return;

    const warning = `WARNING: This will permanently delete selected data segments (${count} selected). Are you absolutely sure?`;
    if (!confirm(warning)) return;

    setIsResetting(true);
    try {
      await invoke("reset_agent", { options: resetOptions });

      if (resetOptions.personality) {
        localStorage.removeItem("pi-hatched");
      }

      alert("Agent reset successful. The application will now reload.");
      window.location.reload();
    } catch (e) {
      console.error("Reset failed:", e);
      alert("Reset failed: " + e);
    } finally {
      setIsResetting(false);
    }
  }

  // Marketplace Logic
  const [marketplaceItems, setMarketplaceItems] = useState<any[]>([]);

  useEffect(() => {
    if (activeTab === "marketplace") {
      invoke("get_mcp_marketplace").then((items: any) =>
        setMarketplaceItems(items),
      );
    }
  }, [activeTab]);

  function handleInstall(item: any) {
    // Pre-fill MCP add form
    setActiveTab("mcp");
    setNewServerName(item.name.toLowerCase().replace(/\s+/g, "-"));
    setNewServerCmd(item.command);
    setNewServerArgs(item.args.join(" "));

    // If there are env vars, show an alert or placeholder
    if (item.env_vars && item.env_vars.length > 0) {
      alert(
        `This server requires the following environment variables: ${item.env_vars.join(", ")}. Please add them to your mcp_config.json manually after adding, or implemented env var UI.`,
      );
      // Note: Current UI doesn't support env vars editing yet.
      // We should probably add that to the Add Server form if we want it to be usable.
      // For now, let's just let them add it and then they can edit config file manually.
    }
  }

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-zinc-900 w-full max-w-2xl rounded-lg shadow-xl overflow-hidden flex flex-col max-h-[80vh]">
        <div className="p-4 border-b border-zinc-200 dark:border-zinc-800 flex justify-between items-center">
          <h2 className="text-xl font-bold dark:text-white">Settings</h2>
          <button
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-700"
          >
            ✕
          </button>
        </div>

        <div className="flex border-b border-zinc-200 dark:border-zinc-800 overflow-x-auto">
          <TabButton
            active={activeTab === "mcp"}
            onClick={() => setActiveTab("mcp")}
          >
            MCP Servers
          </TabButton>
          <TabButton
            active={activeTab === "marketplace"}
            onClick={() => setActiveTab("marketplace")}
          >
            Marketplace
          </TabButton>
          <TabButton
            active={activeTab === "tools"}
            onClick={() => setActiveTab("tools")}
          >
            Tools
          </TabButton>
          <TabButton
            active={activeTab === "models"}
            onClick={() => setActiveTab("models")}
          >
            Models
          </TabButton>
          <TabButton
            active={activeTab === "auth"}
            onClick={() => setActiveTab("auth")}
          >
            Auth
          </TabButton>
          <TabButton
            active={activeTab === "channels"}
            onClick={() => setActiveTab("channels")}
          >
            Channels
          </TabButton>
          <TabButton
            active={activeTab === "agents"}
            onClick={() => setActiveTab("agents")}
          >
            Agents
          </TabButton>
          <TabButton
            active={activeTab === "reset"}
            onClick={() => setActiveTab("reset")}
            className="text-red-500!"
          >
            Reset
          </TabButton>
        </div>

        <div className="p-6 overflow-y-auto flex-1">
          {activeTab === "mcp" && (
            <div className="space-y-6">
              <div className="space-y-4">
                {Object.entries(mcpConfig.mcpServers || {}).map(
                  ([name, conf]: [string, any]) => (
                    <div
                      key={name}
                      className="flex justify-between items-center bg-zinc-50 dark:bg-zinc-800 p-3 rounded"
                    >
                      <div>
                        <div className="font-medium dark:text-white">
                          {name}
                        </div>
                        <div className="text-xs text-zinc-500 font-mono">
                          {conf.command} {conf.args.join(" ")}
                        </div>
                      </div>
                      <button
                        onClick={() => handleRemoveServer(name)}
                        className="text-red-500 hover:text-red-700 text-sm"
                      >
                        Remove
                      </button>
                    </div>
                  ),
                )}
              </div>

              <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
                <h3 className="font-medium mb-2 dark:text-white">Add Server</h3>
                <div className="grid gap-2">
                  <input
                    className="border p-2 rounded dark:bg-zinc-800 dark:text-white"
                    placeholder="Server Name (e.g. git)"
                    value={newServerName}
                    onChange={(e) => setNewServerName(e.target.value)}
                  />
                  <input
                    className="border p-2 rounded dark:bg-zinc-800 dark:text-white"
                    placeholder="Command (e.g. docker)"
                    value={newServerCmd}
                    onChange={(e) => setNewServerCmd(e.target.value)}
                  />
                  <input
                    className="border p-2 rounded dark:bg-zinc-800 dark:text-white"
                    placeholder="Args (space separated)"
                    value={newServerArgs}
                    onChange={(e) => setNewServerArgs(e.target.value)}
                  />
                  <button
                    onClick={handleAddServer}
                    className="bg-blue-600 text-white p-2 rounded hover:bg-blue-700"
                  >
                    Add Server
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "marketplace" && (
            <div className="space-y-4">
              {/* Search Bar & External Links */}
              <div className="flex gap-2">
                <input
                  className="flex-1 border p-2 rounded dark:bg-zinc-800 dark:text-white"
                  placeholder="Search marketplace..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <button
                  onClick={() => open("https://mcpmarket.com")}
                  className="px-3 py-2 bg-zinc-100 dark:bg-zinc-800 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700 text-sm whitespace-nowrap dark:text-gray-300"
                  title="Search on MCPMarket.com"
                >
                  🌍 MCPMarket
                </button>
                <button
                  onClick={() => open("https://mcp.so")}
                  className="px-3 py-2 bg-zinc-100 dark:bg-zinc-800 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700 text-sm whitespace-nowrap dark:text-gray-300"
                  title="Search on mcp.so"
                >
                  🌍 mcp.so
                </button>
              </div>

              {/* Filtered List */}
              <div className="space-y-4">
                {marketplaceItems
                  .filter(
                    (item) =>
                      item.name
                        .toLowerCase()
                        .includes(searchQuery.toLowerCase()) ||
                      item.description
                        .toLowerCase()
                        .includes(searchQuery.toLowerCase()),
                  )
                  .map((item) => (
                    <div
                      key={item.name}
                      className="bg-zinc-50 dark:bg-zinc-800 p-4 rounded flex justify-between items-start gap-4"
                    >
                      <div>
                        <div className="font-bold dark:text-white">
                          {item.name}
                        </div>
                        <div className="text-sm text-zinc-600 dark:text-zinc-400 mb-1">
                          {item.description}
                        </div>
                        <code className="text-xs bg-zinc-200 dark:bg-zinc-950 px-1 rounded">
                          {item.command} {item.args[1]} ...
                        </code>
                      </div>
                      <button
                        onClick={() => handleInstall(item)}
                        className="bg-green-600 text-white px-3 py-1.5 rounded text-sm hover:bg-green-700 whitespace-nowrap"
                      >
                        Install
                      </button>
                    </div>
                  ))}
                {marketplaceItems.length > 0 &&
                  marketplaceItems.filter(
                    (item) =>
                      item.name
                        .toLowerCase()
                        .includes(searchQuery.toLowerCase()) ||
                      item.description
                        .toLowerCase()
                        .includes(searchQuery.toLowerCase()),
                  ).length === 0 && (
                    <div className="text-center py-8 text-zinc-500">
                      No local results found. Try the external search buttons
                      above.
                    </div>
                  )}
              </div>
            </div>
          )}

          {activeTab === "tools" && (
            <div className="space-y-2">
              {/* We don't have a list of ALL tools yet, only enabled config. 
                    Ideally we fetch registry list. For now, let's just show what's in config or hardcoded defaults.
                    Actually, we need a `list_all_tools` command to populate this list. 
                */}
              <div className="text-sm text-zinc-500 mb-4">
                Note: Only shows tools explicitly toggled in config.
              </div>
              {Object.entries(toolsConfig.enabled_tools || {}).map(
                ([name, enabled]: [string, any]) => (
                  <div
                    key={name}
                    className="flex justify-between items-center p-2 rounded bg-zinc-50 dark:bg-zinc-800"
                  >
                    <span className="dark:text-white font-mono">{name}</span>
                    <button
                      onClick={() => handleToggleTool(name, enabled)}
                      className={`px-3 py-1 rounded text-xs ${enabled ? "bg-green-100 text-green-800" : "bg-zinc-200 text-zinc-600"}`}
                    >
                      {enabled ? "Enabled" : "Disabled"}
                    </button>
                  </div>
                ),
              )}
            </div>
          )}

          {activeTab === "models" && (
            <div className="space-y-6">
              {/* Client AI Section */}
              <div className="bg-zinc-100 dark:bg-zinc-800 p-4 rounded border border-zinc-200 dark:border-zinc-700">
                <h3 className="font-bold dark:text-white mb-2">
                  Client-Side AI (Wasm)
                </h3>
                <div className="flex items-center justify-between mb-4">
                  <span className="text-zinc-600 dark:text-zinc-400 text-sm">
                    Status:
                  </span>
                  <span
                    className={`text-sm font-medium ${isClientAILoaded ? "text-green-600 dark:text-green-400" : "text-yellow-600 dark:text-yellow-400"}`}
                  >
                    {isClientAILoaded
                      ? "Loaded & Ready"
                      : "Initializing / Not Loaded"}
                  </span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleTestClientAI}
                    disabled={!isClientAILoaded}
                    className="bg-purple-600 text-white px-3 py-1.5 rounded text-sm hover:bg-purple-700 disabled:opacity-50"
                  >
                    Test Inference
                  </button>
                </div>
                {clientAIPrediction && (
                  <div className="mt-2 p-2 bg-zinc-200 dark:bg-zinc-900 rounded text-xs font-mono dark:text-gray-300 break-all">
                    Result: {clientAIPrediction}
                  </div>
                )}
              </div>

              <div className="space-y-4">
                {localModels.map((m: any) => (
                  <div
                    key={m.model_id}
                    className="flex justify-between items-center bg-zinc-50 dark:bg-zinc-800 p-3 rounded"
                  >
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <div className="font-medium dark:text-white">
                          {m.model_id}
                        </div>
                        {m.loaded && (
                          <span className="text-[10px] bg-green-500 text-white px-1.5 rounded-full">
                            ACTIVE
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-zinc-500">
                        {m.backend} •{" "}
                        {m.downloaded ? "Offline" : "Not Downloaded"}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      {!m.downloaded ? (
                        <button
                          onClick={() => handleDownloadModel(m.model_id)}
                          disabled={downloading[m.model_id]}
                          className="bg-blue-600/10 text-blue-600 dark:bg-blue-900/40 dark:text-blue-400 px-3 py-1 rounded text-sm hover:bg-blue-600/20 disabled:opacity-50"
                        >
                          {downloading[m.model_id]
                            ? "Downloading..."
                            : "Download"}
                        </button>
                      ) : (
                        <>
                          {m.loaded ? (
                            <button
                              onClick={() => handleUnloadModel(m.model_id)}
                              className="bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 px-3 py-1 rounded text-sm hover:bg-red-200"
                            >
                              Unload
                            </button>
                          ) : (
                            <button
                              onClick={() => handleLoadModel(m.model_id)}
                              className="bg-zinc-200 dark:bg-zinc-700 px-3 py-1 rounded text-sm hover:bg-zinc-300 dark:text-white"
                            >
                              Load
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                ))}
                {localModels.length === 0 && (
                  <div className="text-center py-4 text-zinc-500 text-sm italic">
                    Loading model registry...
                  </div>
                )}
              </div>
              <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800 space-y-4">
                <div>
                  <label className="block text-sm font-medium dark:text-zinc-300 mb-1">
                    Local Backend
                  </label>
                  <select
                    style={{
                      backgroundColor: "#18181b",
                      color: "white",
                      backgroundImage: `url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e")`,
                      backgroundRepeat: "no-repeat",
                      backgroundPosition: "right 0.5rem center",
                      backgroundSize: "1.2em 1.2em",
                      paddingRight: "2.5rem",
                    }}
                    className="w-full p-2 rounded border border-blue-500/50 focus:border-blue-400 focus:ring-1 focus:ring-blue-400 text-sm outline-none transition-all cursor-pointer appearance-none"
                    value={localBackend}
                    onChange={(e: any) => setLocalBackend(e.target.value)}
                  >
                    <option value="auto">Auto (Detect by extension)</option>
                    <option value="llama.cpp">llama.cpp (GGUF)</option>
                    <option value="transformers">
                      Transformers (HuggingFace)
                    </option>
                  </select>
                </div>
                <div>
                  <h3 className="font-medium mb-2 dark:text-white">
                    Add Model
                  </h3>
                  <div className="flex gap-2">
                    <input
                      className="flex-1 border p-2 rounded dark:bg-zinc-800 dark:text-white"
                      placeholder="HuggingFace ID or Path"
                      value={newModelId}
                      onChange={(e) => setNewModelId(e.target.value)}
                    />
                    <button
                      onClick={handleAddModel}
                      className="bg-blue-600 text-white px-4 rounded"
                    >
                      Add
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "auth" && (
            <div className="space-y-6">
              <h3 className="font-bold dark:text-white">
                Authentication Providers
              </h3>
              <div className="grid gap-4">
                <div
                  className={`p-4 rounded border-2 ${claudeConnected ? "border-green-500 bg-green-50 dark:bg-green-900/20" : "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800"}`}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="font-bold dark:text-white">
                        Claude Pro/Max
                      </div>
                      <div className="text-sm text-zinc-500">
                        Use your Claude subscription for inference
                      </div>
                      {claudeConnected && (
                        <div className="text-xs text-green-600 dark:text-green-400 mt-1">
                          Connected
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      {claudeConnected ? (
                        <button
                          onClick={handleClaudeDisconnect}
                          className="bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 px-4 py-2 rounded hover:bg-red-200 dark:hover:bg-red-900/50 text-sm"
                        >
                          Disconnect
                        </button>
                      ) : (
                        <button
                          onClick={handleClaudeLogin}
                          disabled={claudeLoading}
                          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                        >
                          {claudeLoading ? "Connecting..." : "Login"}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                <div
                  className={`p-4 rounded border-2 ${antigravityConnected ? "border-green-500 bg-green-50 dark:bg-green-900/20" : "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800"}`}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="font-bold dark:text-white">
                        Google Antigravity
                      </div>
                      <div className="text-sm text-zinc-500">
                        Cloud Code Assist (Internal)
                      </div>
                      {antigravityConnected && (
                        <div className="text-xs text-green-600 dark:text-green-400 mt-1">
                          Connected
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      {antigravityConnected ? (
                        <button
                          onClick={() =>
                            handleProviderDisconnect("antigravity")
                          }
                          className="bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 px-4 py-2 rounded hover:bg-red-200 dark:hover:bg-red-900/50 text-sm"
                        >
                          Disconnect
                        </button>
                      ) : (
                        <button
                          onClick={() => handleLogin("antigravity")}
                          disabled={antigravityLoading}
                          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                        >
                          {antigravityLoading ? "Connecting..." : "Login"}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                <div
                  className={`p-4 rounded border-2 ${geminiConnected ? "border-green-500 bg-green-50 dark:bg-green-900/20" : "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800"}`}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="font-bold dark:text-white">
                        Google Gemini
                      </div>
                      <div className="text-sm text-zinc-500">
                        Standard Generative Language API
                      </div>
                      {geminiConnected && (
                        <div className="text-xs text-green-600 dark:text-green-400 mt-1">
                          Connected
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      {geminiConnected ? (
                        <button
                          onClick={() => handleProviderDisconnect("gemini")}
                          className="bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 px-4 py-2 rounded hover:bg-red-200 dark:hover:bg-red-900/50 text-sm"
                        >
                          Disconnect
                        </button>
                      ) : (
                        <button
                          onClick={() => handleLogin("gemini")}
                          disabled={geminiLoading}
                          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                        >
                          {geminiLoading ? "Connecting..." : "Login"}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {activeTab === "channels" && (
            <div className="space-y-6">
              <h3 className="font-bold dark:text-white">Messaging Channels</h3>

              <div className="p-4 bg-zinc-50 dark:bg-zinc-800 rounded border border-zinc-200 dark:border-zinc-700 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">✈️</span>
                    <div className="font-bold dark:text-white">
                      Telegram Bot
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={telegramConfig.enabled}
                      onChange={async (e) => {
                        const newConfig = {
                          ...telegramConfig,
                          enabled: e.target.checked,
                        };
                        setTelegramConfig(newConfig);
                        try {
                          await invoke("save_telegram_config", {
                            config: newConfig,
                          });
                        } catch (err) {
                          alert("Failed to toggle Telegram: " + err);
                          setTelegramConfig(telegramConfig);
                        }
                      }}
                    />
                    <div className="w-11 h-6 bg-zinc-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-zinc-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="space-y-2">
                  <label className="text-xs font-bold text-zinc-500 uppercase">
                    Bot Token
                  </label>
                  <input
                    type="password"
                    placeholder="Enter bot token from @BotFather"
                    className="w-full p-2 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded text-sm dark:text-white"
                    value={telegramConfig.token || ""}
                    onChange={(e) =>
                      setTelegramConfig({
                        ...telegramConfig,
                        token: e.target.value,
                      })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-xs font-bold text-zinc-500 uppercase">
                    Allowed User IDs (Whitelisted)
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      id="new-tg-user"
                      placeholder="Enter numeric ID..."
                      className="flex-1 p-2 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded text-sm dark:text-white"
                    />
                    <button
                      onClick={() => {
                        const input = document.getElementById(
                          "new-tg-user",
                        ) as HTMLInputElement;
                        const id = parseInt(input.value);
                        if (id && !telegramConfig.allowed_users.includes(id)) {
                          setTelegramConfig({
                            ...telegramConfig,
                            allowed_users: [
                              ...telegramConfig.allowed_users,
                              id,
                            ],
                          });
                          input.value = "";
                        }
                      }}
                      className="bg-blue-600 text-white px-3 rounded text-sm"
                    >
                      Add
                    </button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {telegramConfig.allowed_users.map((id: number) => (
                      <div
                        key={id}
                        className="bg-zinc-200 dark:bg-zinc-700 px-2 py-1 rounded text-xs flex items-center gap-2 dark:text-white"
                      >
                        {id}
                        <button
                          onClick={() =>
                            setTelegramConfig({
                              ...telegramConfig,
                              allowed_users:
                                telegramConfig.allowed_users.filter(
                                  (u: number) => u !== id,
                                ),
                            })
                          }
                          className="text-red-500 font-bold"
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                    {telegramConfig.allowed_users.length === 0 && (
                      <span className="text-xs text-zinc-500 italic">
                        Empty = allow all users (CAUTION)
                      </span>
                    )}
                  </div>
                </div>

                <button
                  disabled={telegramLoading}
                  onClick={async () => {
                    setTelegramLoading(true);
                    try {
                      await invoke("save_telegram_config", {
                        config: telegramConfig,
                      });
                      alert("Telegram configuration saved!");
                    } catch (err) {
                      alert("Failed to save config: " + err);
                    } finally {
                      setTelegramLoading(false);
                    }
                  }}
                  className="w-full py-2 bg-blue-600 text-white rounded font-bold hover:bg-blue-700 disabled:opacity-50"
                >
                  {telegramLoading ? "Saving..." : "Apply & Save Config"}
                </button>
              </div>

              <div className="p-4 bg-zinc-50 dark:bg-zinc-800 rounded border border-zinc-200 dark:border-zinc-700 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">💬</span>
                    <div className="font-bold dark:text-white">Discord Bot</div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={discordConfig.enabled}
                      onChange={async (e) => {
                        const newConfig = {
                          ...discordConfig,
                          enabled: e.target.checked,
                        };
                        setDiscordConfig(newConfig);
                        try {
                          await invoke("save_discord_config", {
                            config: newConfig,
                          });
                        } catch (err) {
                          alert("Failed to toggle Discord: " + err);
                          setDiscordConfig(discordConfig);
                        }
                      }}
                    />
                    <div className="w-11 h-6 bg-zinc-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-zinc-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                <div className="space-y-2">
                  <label className="text-xs font-bold text-zinc-500 uppercase">
                    Bot Token
                  </label>
                  <input
                    type="password"
                    placeholder="Enter bot token from Discord Developer Portal"
                    className="w-full p-2 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded text-sm dark:text-white"
                    value={discordConfig.token || ""}
                    onChange={(e) =>
                      setDiscordConfig({
                        ...discordConfig,
                        token: e.target.value,
                      })
                    }
                  />
                </div>

                <button
                  disabled={discordLoading}
                  onClick={async () => {
                    setDiscordLoading(true);
                    try {
                      await invoke("save_discord_config", {
                        config: discordConfig,
                      });
                      alert("Discord configuration saved!");
                    } catch (err) {
                      alert("Failed to save config: " + err);
                    } finally {
                      setDiscordLoading(false);
                    }
                  }}
                  className="w-full py-2 bg-blue-600 text-white rounded font-bold hover:bg-blue-700 disabled:opacity-50"
                >
                  {discordLoading ? "Saving..." : "Apply & Save Config"}
                </button>
              </div>
            </div>
          )}

          {activeTab === "agents" && (
            <div className="space-y-6">
              <h3 className="font-bold dark:text-white">Multi-Agent Routing</h3>
              <p className="text-sm text-zinc-500">
                Create multiple agent instances and route messaging channels to
                specific agents.
              </p>

              <div className="p-4 bg-zinc-50 dark:bg-zinc-800 rounded border border-zinc-200 dark:border-zinc-700 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="font-medium dark:text-white">
                    🤖 Default Agent
                  </span>
                  <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 px-2 py-1 rounded">
                    Active
                  </span>
                </div>
                <div className="text-xs text-zinc-500">
                  The default agent handles all unrouted channels and direct
                  messages.
                </div>
              </div>

              <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
                <h4 className="font-medium mb-3 dark:text-white">
                  Channel Routing
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between p-2 bg-zinc-50 dark:bg-zinc-800 rounded">
                    <span className="dark:text-white">✈️ Telegram</span>
                    <span className="text-zinc-500">→ Default Agent</span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-zinc-50 dark:bg-zinc-800 rounded">
                    <span className="dark:text-white">💬 Discord</span>
                    <span className="text-zinc-500">→ Default Agent</span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  💡 <strong>Coming Soon:</strong> Use the <code>sessions</code>{" "}
                  tool to create additional agents and configure routing
                  dynamically.
                </p>
              </div>
            </div>
          )}

          {activeTab === "reset" && (
            <div className="space-y-6">
              <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <h3 className="text-red-800 dark:text-red-400 font-bold mb-1">
                  Danger Zone
                </h3>
                <p className="text-sm text-red-700 dark:text-red-300">
                  Resetting the agent will permanently delete the selected
                  configurations and data. This action cannot be undone.
                </p>
              </div>

              <div className="space-y-3">
                <ResetOption
                  label="Clear Memory & History"
                  description="Deletes all conversation logs and knowledge base (memory.db)"
                  checked={resetOptions.memory}
                  onChange={(v) =>
                    setResetOptions({ ...resetOptions, memory: v })
                  }
                />
                <ResetOption
                  label="Reset Personality & Hatching"
                  description="Allows you to experience the 'hatching' sequence again"
                  checked={resetOptions.personality}
                  onChange={(v) =>
                    setResetOptions({ ...resetOptions, personality: v })
                  }
                />
                <ResetOption
                  label="Reset MCP Servers"
                  description="Removes all installed MCP server configurations"
                  checked={resetOptions.mcp_config}
                  onChange={(v) =>
                    setResetOptions({ ...resetOptions, mcp_config: v })
                  }
                />
                <ResetOption
                  label="Reset Skills/Tools"
                  description="Reverts tool enablement settings to defaults"
                  checked={resetOptions.tools_config}
                  onChange={(v) =>
                    setResetOptions({ ...resetOptions, tools_config: v })
                  }
                />
                <ResetOption
                  label="Reset Model Config"
                  description="Removes custom added models from the list"
                  checked={resetOptions.models_config}
                  onChange={(v) =>
                    setResetOptions({ ...resetOptions, models_config: v })
                  }
                />
              </div>

              <div className="pt-4">
                <button
                  onClick={handleReset}
                  disabled={
                    isResetting || !Object.values(resetOptions).some(Boolean)
                  }
                  className="w-full py-3 bg-red-600 hover:bg-red-700 disabled:bg-zinc-400 text-white font-bold rounded-lg shadow transition-colors"
                >
                  {isResetting ? "Resetting..." : "Execute Reset"}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ResetOption({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div
      className="flex items-start gap-3 p-3 bg-zinc-50 dark:bg-zinc-800 rounded-lg border border-transparent hover:border-zinc-200 dark:hover:border-zinc-700 transition-colors cursor-pointer"
      onClick={() => onChange(!checked)}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 h-4 w-4 rounded border-gray-300 text-red-600 focus:ring-red-600"
        onClick={(e) => e.stopPropagation()}
      />
      <div className="flex-1">
        <div className="text-sm font-medium dark:text-white leading-none">
          {label}
        </div>
        <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
          {description}
        </div>
      </div>
    </div>
  );
}

function TabButton({ children, active, onClick }: any) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 py-3 text-sm font-medium border-b-2 transition-colors ${active ? "border-blue-600 text-blue-600" : "border-transparent text-zinc-500 hover:text-zinc-700"}`}
    >
      {children}
    </button>
  );
}
