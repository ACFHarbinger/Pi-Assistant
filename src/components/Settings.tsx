import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-shell';

interface McpServerConfig {
    command: string;
    args: string[];
    env: Record<string, string>;
}

interface ToolsConfig {
    enabled_tools: Record<string, boolean>;
}

interface ModelInfo {
    id: string;
    path?: string;
    description?: string;
}

interface ModelsConfig {
    models: ModelInfo[];
}

export default function Settings({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
    const [activeTab, setActiveTab] = useState<'mcp' | 'tools' | 'models' | 'marketplace'>('mcp');
    const [mcpConfig, setMcpConfig] = useState<any>({});
    const [toolsConfig, setToolsConfig] = useState<any>({});
    const [modelsConfig, setModelsConfig] = useState<any>({ models: [] });
    // Marketplace Search
    const [searchQuery, setSearchQuery] = useState('');

    // Forms
    const [newServerName, setNewServerName] = useState('');
    const [newServerCmd, setNewServerCmd] = useState('');
    const [newServerArgs, setNewServerArgs] = useState('');

    const [newModelId, setNewModelId] = useState('');

    useEffect(() => {
        if (isOpen) {
            refreshConfig();
        }
    }, [isOpen]);

    async function refreshConfig() {
        try {
            const mcp = await invoke('get_mcp_config');
            setMcpConfig(mcp);
            const tools = await invoke('get_tools_config');
            setToolsConfig(tools);
            const models = await invoke('get_models_config');
            setModelsConfig(models);
        } catch (e) {
            console.error('Failed to load config:', e);
        }
    }

    async function handleAddServer() {
        try {
            const args = newServerArgs.split(' ').filter(s => s.trim().length > 0);
            await invoke('save_mcp_server', {
                name: newServerName,
                config: { command: newServerCmd, args, env: {} }
            });
            setNewServerName('');
            setNewServerCmd('');
            setNewServerArgs('');
            refreshConfig();
        } catch (e) {
            console.error(e);
            alert('Failed to save server');
        }
    }

    async function handleRemoveServer(name: string) {
        if (!confirm(`Remove server ${name}?`)) return;
        await invoke('remove_mcp_server', { name });
        refreshConfig();
    }

    async function handleToggleTool(name: string, current: boolean) {
        await invoke('toggle_tool', { name, enabled: !current });
        refreshConfig();
    }

    async function handleAddModel() {
        await invoke('save_model', {
            model: { id: newModelId, description: 'User added' }
        });
        setNewModelId('');
        refreshConfig();
    }

    async function handleLoadModel(id: string) {
        try {
            await invoke('load_model', { modelId: id });
            alert('Model loaded!');
        } catch (e) {
            console.error(e);
            alert('Failed to load model: ' + e);
        }
    }

    // Marketplace Logic
    const [marketplaceItems, setMarketplaceItems] = useState<any[]>([]);

    useEffect(() => {
        if (activeTab === 'marketplace') {
            invoke('get_mcp_marketplace').then((items: any) => setMarketplaceItems(items));
        }
    }, [activeTab]);

    function handleInstall(item: any) {
        // Pre-fill MCP add form
        setActiveTab('mcp');
        setNewServerName(item.name.toLowerCase().replace(/\s+/g, '-'));
        setNewServerCmd(item.command);
        setNewServerArgs(item.args.join(' '));

        // If there are env vars, show an alert or placeholder
        if (item.env_vars && item.env_vars.length > 0) {
            alert(`This server requires the following environment variables: ${item.env_vars.join(', ')}. Please add them to your mcp_config.json manually after adding, or implemented env var UI.`);
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
                    <button onClick={onClose} className="text-zinc-500 hover:text-zinc-700">✕</button>
                </div>

                <div className="flex border-b border-zinc-200 dark:border-zinc-800 overflow-x-auto">
                    <TabButton active={activeTab === 'mcp'} onClick={() => setActiveTab('mcp')}>MCP Servers</TabButton>
                    <TabButton active={activeTab === 'marketplace'} onClick={() => setActiveTab('marketplace')}>Marketplace</TabButton>
                    <TabButton active={activeTab === 'tools'} onClick={() => setActiveTab('tools')}>Tools</TabButton>
                    <TabButton active={activeTab === 'models'} onClick={() => setActiveTab('models')}>Models</TabButton>
                </div>

                <div className="p-6 overflow-y-auto flex-1">
                    {activeTab === 'mcp' && (
                        <div className="space-y-6">
                            <div className="space-y-4">
                                {Object.entries(mcpConfig.mcpServers || {}).map(([name, conf]: [string, any]) => (
                                    <div key={name} className="flex justify-between items-center bg-zinc-50 dark:bg-zinc-800 p-3 rounded">
                                        <div>
                                            <div className="font-medium dark:text-white">{name}</div>
                                            <div className="text-xs text-zinc-500 font-mono">{conf.command} {conf.args.join(' ')}</div>
                                        </div>
                                        <button onClick={() => handleRemoveServer(name)} className="text-red-500 hover:text-red-700 text-sm">Remove</button>
                                    </div>
                                ))}
                            </div>

                            <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
                                <h3 className="font-medium mb-2 dark:text-white">Add Server</h3>
                                <div className="grid gap-2">
                                    <input className="border p-2 rounded dark:bg-zinc-800 dark:text-white" placeholder="Server Name (e.g. git)" value={newServerName} onChange={e => setNewServerName(e.target.value)} />
                                    <input className="border p-2 rounded dark:bg-zinc-800 dark:text-white" placeholder="Command (e.g. docker)" value={newServerCmd} onChange={e => setNewServerCmd(e.target.value)} />
                                    <input className="border p-2 rounded dark:bg-zinc-800 dark:text-white" placeholder="Args (space separated)" value={newServerArgs} onChange={e => setNewServerArgs(e.target.value)} />
                                    <button onClick={handleAddServer} className="bg-blue-600 text-white p-2 rounded hover:bg-blue-700">Add Server</button>
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === 'marketplace' && (
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
                                    onClick={() => open('https://mcpmarket.com')}
                                    className="px-3 py-2 bg-zinc-100 dark:bg-zinc-800 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700 text-sm whitespace-nowrap dark:text-gray-300"
                                    title="Search on MCPMarket.com"
                                >
                                    🌍 MCPMarket
                                </button>
                                <button
                                    onClick={() => open('https://mcp.so')}
                                    className="px-3 py-2 bg-zinc-100 dark:bg-zinc-800 rounded hover:bg-zinc-200 dark:hover:bg-zinc-700 text-sm whitespace-nowrap dark:text-gray-300"
                                    title="Search on mcp.so"
                                >
                                    🌍 mcp.so
                                </button>
                            </div>

                            {/* Filtered List */}
                            <div className="space-y-4">
                                {marketplaceItems
                                    .filter(item =>
                                        item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                                        item.description.toLowerCase().includes(searchQuery.toLowerCase())
                                    )
                                    .map((item) => (
                                        <div key={item.name} className="bg-zinc-50 dark:bg-zinc-800 p-4 rounded flex justify-between items-start gap-4">
                                            <div>
                                                <div className="font-bold dark:text-white">{item.name}</div>
                                                <div className="text-sm text-zinc-600 dark:text-zinc-400 mb-1">{item.description}</div>
                                                <code className="text-xs bg-zinc-200 dark:bg-zinc-950 px-1 rounded">{item.command} {item.args[1]} ...</code>
                                            </div>
                                            <button
                                                onClick={() => handleInstall(item)}
                                                className="bg-green-600 text-white px-3 py-1.5 rounded text-sm hover:bg-green-700 whitespace-nowrap"
                                            >
                                                Install
                                            </button>
                                        </div>
                                    ))}
                                {marketplaceItems.length > 0 && marketplaceItems.filter(item =>
                                    item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                                    item.description.toLowerCase().includes(searchQuery.toLowerCase())
                                ).length === 0 && (
                                        <div className="text-center py-8 text-zinc-500">
                                            No local results found. Try the external search buttons above.
                                        </div>
                                    )}
                            </div>
                        </div>
                    )}

                    {activeTab === 'tools' && (
                        <div className="space-y-2">
                            {/* We don't have a list of ALL tools yet, only enabled config. 
                    Ideally we fetch registry list. For now, let's just show what's in config or hardcoded defaults.
                    Actually, we need a `list_all_tools` command to populate this list. 
                */}
                            <div className="text-sm text-zinc-500 mb-4">
                                Note: Only shows tools explicitly toggled in config.
                            </div>
                            {Object.entries(toolsConfig.enabled_tools || {}).map(([name, enabled]: [string, any]) => (
                                <div key={name} className="flex justify-between items-center p-2 rounded bg-zinc-50 dark:bg-zinc-800">
                                    <span className="dark:text-white font-mono">{name}</span>
                                    <button
                                        onClick={() => handleToggleTool(name, enabled)}
                                        className={`px-3 py-1 rounded text-xs ${enabled ? 'bg-green-100 text-green-800' : 'bg-zinc-200 text-zinc-600'}`}
                                    >
                                        {enabled ? 'Enabled' : 'Disabled'}
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    {activeTab === 'models' && (
                        <div className="space-y-6">
                            <div className="space-y-4">
                                {modelsConfig.models.map((m: any) => (
                                    <div key={m.id} className="flex justify-between items-center bg-zinc-50 dark:bg-zinc-800 p-3 rounded">
                                        <div>
                                            <div className="font-medium dark:text-white">{m.id}</div>
                                            {m.description && <div className="text-xs text-zinc-500">{m.description}</div>}
                                        </div>
                                        <button onClick={() => handleLoadModel(m.id)} className="bg-zinc-200 dark:bg-zinc-700 px-3 py-1 rounded text-sm hover:bg-zinc-300 dark:hover:bg-zinc-600 dark:text-white">Load</button>
                                    </div>
                                ))}
                            </div>
                            <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800">
                                <h3 className="font-medium mb-2 dark:text-white">Add Model</h3>
                                <div className="flex gap-2">
                                    <input className="flex-1 border p-2 rounded dark:bg-zinc-800 dark:text-white" placeholder="HuggingFace ID or Path" value={newModelId} onChange={e => setNewModelId(e.target.value)} />
                                    <button onClick={handleAddModel} className="bg-blue-600 text-white px-4 rounded">Add</button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function TabButton({ children, active, onClick }: any) {
    return (
        <button
            onClick={onClick}
            className={`flex-1 py-3 text-sm font-medium border-b-2 transition-colors ${active ? 'border-blue-600 text-blue-600' : 'border-transparent text-zinc-500 hover:text-zinc-700'}`}
        >
            {children}
        </button>
    )
}
