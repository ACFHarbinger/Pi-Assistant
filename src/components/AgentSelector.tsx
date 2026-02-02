import React from "react";
import { useAgentStore } from "../stores/agentStore";

export const AgentSelector: React.FC = () => {
  const { agents, activeAgentId, setActiveAgent, stopAgent } = useAgentStore();

  const agentList = Object.values(agents);

  if (agentList.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-2 p-2 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
      <div className="text-xs font-semibold text-gray-500 uppercase">
        Active Agents
      </div>
      <div className="flex flex-wrap gap-2">
        {agentList.map((agent) => {
          const agentId = agent.data?.agent_id;
          if (!agentId) return null;

          const isActive = agentId === activeAgentId;
          const status = agent.status;

          // Truncate task description or use ID
          const label =
            agent.data?.task_tree?.[0]?.title ||
            agent.data?.question?.slice(0, 20) ||
            `Agent ${agentId.slice(0, 8)}`;

          return (
            <div
              key={agentId}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-full text-sm border cursor-pointer transition-colors
                ${
                  isActive
                    ? "bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300"
                    : "bg-white border-gray-300 text-gray-700 hover:bg-gray-100 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 hover:dark:bg-gray-700"
                }
              `}
              onClick={() => setActiveAgent(agentId)}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  status === "Running"
                    ? "bg-green-500 animate-pulse"
                    : status === "Paused"
                      ? "bg-yellow-500"
                      : status === "Stopped"
                        ? "bg-red-500"
                        : "bg-gray-400"
                }`}
              />

              <span className="max-w-[150px] truncate">{label}</span>

              <button
                className="ml-1 p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10"
                onClick={(e) => {
                  e.stopPropagation();
                  stopAgent(agentId);
                }}
                title="Stop Agent"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="w-4 h-4"
                >
                  <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
                </svg>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
