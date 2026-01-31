import { useState, useRef, useEffect } from "react";
import { useAgentStore } from "../stores/agentStore";

export function ChatInterface() {
    const { messages, state, sendMessage } = useAgentStore();
    const [input, setInput] = useState("");
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;
        sendMessage(input.trim());
        setInput("");
    };

    const isWaitingForInput = state.status === "Paused" && state.data?.question;

    return (
        <div className="glass rounded-2xl h-[600px] flex flex-col">
            {/* Header */}
            <div className="px-6 py-4 border-b border-white/10">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                    <span className="text-primary-400">💬</span>
                    Chat
                </h2>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-gray-500">
                        <div className="text-center">
                            <p className="text-4xl mb-2">🤖</p>
                            <p>No messages yet. Start a task to begin!</p>
                        </div>
                    </div>
                ) : (
                    messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                            <div
                                className={`max-w-[80%] rounded-2xl px-4 py-3 ${msg.role === "user"
                                        ? "bg-primary-600 text-white"
                                        : msg.role === "system"
                                            ? "bg-gray-800 text-gray-300 text-sm italic"
                                            : "bg-gray-800 text-gray-100"
                                    }`}
                            >
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                                <p className="text-xs opacity-50 mt-1">
                                    {msg.timestamp.toLocaleTimeString()}
                                </p>
                            </div>
                        </div>
                    ))
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Question Banner */}
            {isWaitingForInput && (
                <div className="mx-6 mb-4 p-4 rounded-xl bg-yellow-500/20 border border-yellow-500/30">
                    <p className="text-yellow-300 text-sm font-medium">
                        🤔 Agent is asking: {state.data?.question}
                    </p>
                </div>
            )}

            {/* Input */}
            <form onSubmit={handleSubmit} className="p-4 border-t border-white/10">
                <div className="flex gap-3">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={isWaitingForInput ? "Type your answer..." : "Send a message..."}
                        className="flex-1 bg-gray-800 rounded-xl px-4 py-3 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
                    />
                    <button
                        type="submit"
                        disabled={!input.trim()}
                        className="px-6 py-3 rounded-xl bg-primary-600 text-white font-medium text-sm hover:bg-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        Send
                    </button>
                </div>
            </form>
        </div>
    );
}
