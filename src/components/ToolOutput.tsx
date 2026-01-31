interface ToolOutputProps {
  toolName: string;
  command?: string;
  output: string;
  error?: string;
  success: boolean;
  timestamp?: string;
}

export function ToolOutput({
  toolName,
  command,
  output,
  error,
  success,
  timestamp,
}: ToolOutputProps) {
  const getToolIcon = (name: string) => {
    switch (name) {
      case "shell":
        return "💻";
      case "code":
        return "📝";
      case "browser":
        return "🌐";
      default:
        return "🔧";
    }
  };

  return (
    <div
      className={`glass rounded-xl p-4 border ${success ? "border-green-500/20" : "border-red-500/20"}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">{getToolIcon(toolName)}</span>
          <span className="font-semibold text-sm">{toolName}</span>
          <span
            className={`px-2 py-0.5 rounded-full text-xs ${success ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}
          >
            {success ? "Success" : "Failed"}
          </span>
        </div>
        {timestamp && (
          <span className="text-xs text-gray-500">{timestamp}</span>
        )}
      </div>

      {/* Command */}
      {command && (
        <div className="mb-3">
          <span className="text-xs text-gray-400 block mb-1">Command</span>
          <code className="block text-sm bg-gray-900/50 rounded-lg p-2 font-mono text-primary-300 overflow-x-auto">
            {command}
          </code>
        </div>
      )}

      {/* Output */}
      <div>
        <span className="text-xs text-gray-400 block mb-1">Output</span>
        <pre className="text-sm bg-gray-900/50 rounded-lg p-3 font-mono text-gray-300 overflow-x-auto max-h-48 overflow-y-auto whitespace-pre-wrap">
          {output || "(no output)"}
        </pre>
      </div>

      {/* Error */}
      {error && (
        <div className="mt-3">
          <span className="text-xs text-red-400 block mb-1">Error</span>
          <pre className="text-sm bg-red-900/20 rounded-lg p-3 font-mono text-red-300 overflow-x-auto">
            {error}
          </pre>
        </div>
      )}
    </div>
  );
}

// List of tool outputs
interface ToolOutputListProps {
  outputs: ToolOutputProps[];
}

export function ToolOutputList({ outputs }: ToolOutputListProps) {
  if (outputs.length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        <span className="text-4xl block mb-2">🔧</span>
        <p>No tool executions yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {outputs.map((output, index) => (
        <ToolOutput key={index} {...output} />
      ))}
    </div>
  );
}
