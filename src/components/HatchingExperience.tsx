import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';

interface HatchingExperienceProps {
    onComplete: () => void;
}

// Simple markdown-like formatting for the hatching message
function formatMessage(text: string): JSX.Element[] {
    return text.split('\n').map((line, i) => {
        // Bold: **text**
        const formatted = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        return (
            <p
                key={i}
                className={line.startsWith('-') ? 'ml-4' : ''}
                dangerouslySetInnerHTML={{ __html: formatted || '&nbsp;' }}
            />
        );
    });
}

export function HatchingExperience({ onComplete }: HatchingExperienceProps) {
    const [message, setMessage] = useState<string>('');
    const [visible, setVisible] = useState(false);
    const [showContinue, setShowContinue] = useState(false);

    useEffect(() => {
        let isStopped = false;

        async function fetchHatchingMessage() {
            console.log('HatchingExperience: Fetching message...');
            try {
                const result = await invoke<{ message: string }>('sidecar_request', {
                    method: 'personality.get_hatching',
                    params: {},
                });
                if (isStopped) return;
                console.log('HatchingExperience: Received message:', result);
                setMessage(result.message);
                setVisible(true);
            } catch (e) {
                if (isStopped) return;
                console.error('HatchingExperience: Failed to fetch message:', e);
                setMessage("Hi! I'm Pi, your personal AI assistant. 🎉\n\nReady to get started?");
                setVisible(true);
            }
        }

        fetchHatchingMessage();

        // Safety timeout (5s) to force visibility if sidecar hangs
        const timeout = setTimeout(() => {
            if (!isStopped && !visible) {
                console.warn('HatchingExperience: Sidecar timeout, forcing visibility');
                setMessage("Hey there! I'm Pi. Ready to help you with anything you need. 🤖");
                setVisible(true);
                setShowContinue(true);
            }
        }, 5000);

        return () => {
            isStopped = true;
            clearTimeout(timeout);
        };
    }, []); // Run once on mount

    const handleContinue = () => {
        // Mark as hatched in localStorage
        localStorage.setItem('pi-hatched', 'true');
        onComplete();
    };

    return (
        <div className="fixed inset-0 z-50 bg-gray-950 flex flex-col items-center justify-center p-6">
            {!visible && (
                <div className="absolute inset-0 flex items-center justify-center z-0">
                    <div className="text-primary-500 animate-spin text-4xl">⏳</div>
                    <p className="ml-4 text-gray-400 animate-pulse">Waking up Pi...</p>
                </div>
            )}
            <div
                className={`relative z-10 flex flex-col items-center max-w-lg transition-all duration-1000 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
                    }`}
            >
                {/* Animated Pi Icon */}
                <div className="flex justify-center mb-8">
                    <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center animate-pulse shadow-lg shadow-primary-500/30">
                        <span className="text-5xl">🤖</span>
                    </div>
                </div>

                {/* Hatching Message */}
                <div className="glass rounded-2xl p-8 mb-8 text-center bg-white/5 border border-white/10 shadow-2xl">
                    <div className="space-y-4 text-white text-lg leading-relaxed">
                        {formatMessage(message)}
                    </div>
                </div>

                {/* Continue button appears after delay or message load */}
                <div className={`transition-all duration-700 ${showContinue || message.length > 0 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                    <button
                        onClick={handleContinue}
                        className="px-10 py-4 bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-500 hover:to-accent-500 text-white font-bold rounded-xl shadow-lg shadow-primary-900/20 transition-all hover:scale-105 active:scale-95"
                    >
                        Begin Journey
                    </button>
                </div>
            </div>
        </div>
    );
}
