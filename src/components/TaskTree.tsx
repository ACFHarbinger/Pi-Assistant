import React, { useMemo } from "react";
import {
  CheckCircle2,
  Circle,
  Loader2,
  XCircle,
  AlertCircle,
} from "lucide-react";

interface TaskTreeProps {
  subtasks: any[]; // Use any or specific Subtask type from store
  activeSubtaskId?: string | null;
}

export const TaskTree: React.FC<TaskTreeProps> = ({
  subtasks,
  activeSubtaskId,
}) => {
  // Build the tree structure locally
  const tree = useMemo(() => {
    const map = new Map<string, any>();
    const root: any[] = [];

    subtasks.forEach((task) => {
      map.set(task.id, { ...task, children: [] });
    });

    subtasks.forEach((task) => {
      if (task.parent_id && map.has(task.parent_id)) {
        map.get(task.parent_id).children.push(map.get(task.id));
      } else {
        root.push(map.get(task.id));
      }
    });

    return root;
  }, [subtasks]);

  const renderStatusIcon = (status: string, isActive: boolean) => {
    if (isActive || status === "running") {
      return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />;
    }
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-4 h-4 text-green-400" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-red-400" />;
      case "blocked":
        return <AlertCircle className="w-4 h-4 text-orange-400" />;
      default:
        return <Circle className="w-4 h-4 text-slate-500" />;
    }
  };

  const TaskNode = ({ node, depth }: { node: any; depth: number }) => (
    <div className="flex flex-col">
      <div
        className={`flex items-center gap-3 p-2 rounded-lg transition-colors ${
          node.id === activeSubtaskId
            ? "bg-blue-500/10 border border-blue-500/20"
            : "hover:bg-white/5"
        }`}
        style={{ marginLeft: `${depth * 1.5}rem` }}
      >
        <div className="mt-0.5">
          {renderStatusIcon(node.status, node.id === activeSubtaskId)}
        </div>
        <div className="flex flex-col">
          <span
            className={`text-sm font-medium ${
              node.status === "completed"
                ? "text-slate-400 line-through"
                : "text-slate-200"
            }`}
          >
            {node.title}
          </span>
          {node.description && (
            <span className="text-xs text-slate-500 truncate max-w-[200px]">
              {node.description}
            </span>
          )}
        </div>
      </div>
      {node.children.map((child: any) => (
        <TaskNode key={child.id} node={child} depth={depth + 1} />
      ))}
    </div>
  );

  return (
    <div className="flex flex-col gap-1 overflow-y-auto max-h-[300px] pr-2 custom-scrollbar">
      {tree.length === 0 ? (
        <div className="text-sm text-slate-500 italic p-4 text-center">
          No active subtasks.
        </div>
      ) : (
        tree.map((node) => <TaskNode key={node.id} node={node} depth={0} />)
      )}
    </div>
  );
};
