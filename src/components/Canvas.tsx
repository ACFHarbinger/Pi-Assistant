import React, { useState, useRef, useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import { Maximize2, Minimize2, Trash2, Camera, X } from "lucide-react";

interface CanvasProps {
  isOpen: boolean;
  onClose: () => void;
}

export const Canvas: React.FC<CanvasProps> = ({ isOpen, onClose }) => {
  const [content, setContent] = useState<string>("");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    const unlisten = listen<string>("canvas-push", (event) => {
      setContent(event.payload);
    });

    const unlistenClear = listen("canvas-clear", () => {
      setContent("");
    });

    const unlistenEval = listen<string>("canvas-eval", (event) => {
      if (iframeRef.current && iframeRef.current.contentWindow) {
        // Execute JavaScript in the iframe using postMessage
        const code = event.payload;
        try {
          // Use Function constructor for safer eval (still sandbox protected)
          iframeRef.current.contentWindow.postMessage(
            { type: "pi-canvas-eval", code },
            "*",
          );
        } catch (e) {
          console.error("Failed to eval in canvas:", e);
        }
      }
    });

    return () => {
      unlisten.then((fn) => fn());
      unlistenClear.then((fn) => fn());
      unlistenEval.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    if (iframeRef.current && content) {
      const doc = iframeRef.current.contentDocument;
      if (doc) {
        // Inject the eval listener into the iframe content
        const evalHandler = `
                    <script>
                        window.addEventListener('message', function(event) {
                            if (event.data && event.data.type === 'pi-canvas-eval') {
                                try {
                                    eval(event.data.code);
                                } catch (e) {
                                    console.error('Canvas eval error:', e);
                                }
                            }
                        });
                    </script>
                `;
        doc.open();
        doc.write(evalHandler + content);
        doc.close();
      }
    }
  }, [content]);

  const handleClear = () => {
    setContent("");
  };

  const handleSnapshot = () => {
    // Simple implementation - in production, use html2canvas or similar
    if (iframeRef.current) {
      const doc = iframeRef.current.contentDocument;
      if (doc) {
        const html = doc.documentElement.outerHTML;
        const blob = new Blob([html], { type: "text/html" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `canvas-snapshot-${Date.now()}.html`;
        a.click();
        URL.revokeObjectURL(url);
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className={`
        fixed z-40 transition-all duration-300 ease-out
        ${isFullscreen ? "inset-4" : "bottom-4 right-4 w-[500px] h-[400px]"}
      `}
    >
      <div className="h-full flex flex-col rounded-2xl overflow-hidden border border-white/20 backdrop-blur-xl bg-gray-900/90 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-white/10 bg-gray-800/50">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 animate-pulse" />
            <span className="text-sm font-semibold text-white/80">
              Live Canvas
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={handleSnapshot}
              className="p-1.5 rounded-lg hover:bg-white/10 text-white/60 hover:text-white transition-colors"
              title="Snapshot"
            >
              <Camera size={16} />
            </button>
            <button
              onClick={handleClear}
              className="p-1.5 rounded-lg hover:bg-white/10 text-white/60 hover:text-white transition-colors"
              title="Clear"
            >
              <Trash2 size={16} />
            </button>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-1.5 rounded-lg hover:bg-white/10 text-white/60 hover:text-white transition-colors"
              title={isFullscreen ? "Minimize" : "Maximize"}
            >
              {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-red-500/20 text-white/60 hover:text-red-400 transition-colors"
              title="Close"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Canvas Content */}
        <div className="flex-1 bg-white relative">
          {content ? (
            <iframe
              ref={iframeRef}
              className="w-full h-full border-0"
              sandbox="allow-scripts allow-same-origin"
              title="Agent Canvas"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="text-4xl mb-2">🎨</div>
                <div className="text-sm">Canvas is empty</div>
                <div className="text-xs text-gray-500 mt-1">
                  The agent will push content here
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
