import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';

interface HatchingExperienceProps {
    onComplete: () => void;
}

type WizardStep = 'welcome' | 'model' | 'api-key' | 'skills' | 'hatching';

export function HatchingExperience({ onComplete }: HatchingExperienceProps) {
    const [step, setStep] = useState<WizardStep>('welcome');
    const [provider, setProvider] = useState<'anthropic' | 'gemini' | 'local'>('anthropic');
    const [apiKey, setApiKey] = useState('');
    const [skills, setSkills] = useState({
        shell: true,
        browser: true,
        files: true
    });
    const [hatchingMessage, setHatchingMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showAdvancedAuth, setShowAdvancedAuth] = useState(false);

    // Fetch hatching message from sidecar for the final step
    useEffect(() => {
        if (step === 'hatching') {
            const fetchHatchingMessage = async () => {
                try {
                    const result = await invoke<{ message: string }>('sidecar_request', {
                        method: 'personality.get_hatching',
                        params: {},
                    });
                    setHatchingMessage(result.message);
                } catch (e) {
                    console.error('Failed to fetch hatching message:', e);
                    setHatchingMessage("I'm Pi, your personal AI assistant. Let's get started!");
                }
            };
            fetchHatchingMessage();
        }
    }, [step]);

    const handleNext = async () => {
        if (step === 'welcome') setStep('model');
        else if (step === 'model') {
            if (provider === 'local') setStep('skills');
            else setStep('api-key');
        }
        else if (step === 'api-key') {
            if (!apiKey) {
                alert('Please enter an API key to continue.');
                return;
            }
            setIsLoading(true);
            try {
                await invoke('save_api_key', { provider, key: apiKey });
                setStep('skills');
            } catch (e) {
                alert('Failed to save API key: ' + e);
            } finally {
                setIsLoading(false);
            }
        }
        else if (step === 'skills') {
            setIsLoading(true);
            try {
                // Save tool enablement
                await invoke('toggle_tool', { name: 'shell', enabled: skills.shell });
                await invoke('toggle_tool', { name: 'browser', enabled: skills.browser });
                await invoke('toggle_tool', { name: 'filesystem', enabled: skills.files });
                setStep('hatching');
            } catch (e) {
                console.error('Failed to save skills:', e);
                setStep('hatching'); // Proceed anyway
            } finally {
                setIsLoading(false);
            }
        }
    };

    const handleComplete = () => {
        localStorage.setItem('pi-hatched', 'true');
        onComplete();
    };

    return (
        <div className="fixed inset-0 z-50 bg-gray-950 flex items-center justify-center p-6 font-sans">
            <div className="max-w-xl w-full">
                {/* Progress Indicators */}
                <div className="flex justify-center gap-2 mb-12">
                    {(['welcome', 'model', 'api-key', 'skills', 'hatching'] as WizardStep[]).map((s, i) => (
                        <div
                            key={s}
                            className={`h-1.5 rounded-full transition-all duration-500 ${step === s ? 'w-8 bg-primary-500' :
                                i < ['welcome', 'model', 'api-key', 'skills', 'hatching'].indexOf(step) ? 'w-4 bg-primary-800' : 'w-4 bg-gray-800'
                                }`}
                        />
                    ))}
                </div>

                <div className="glass rounded-3xl p-10 bg-white/5 border border-white/10 shadow-2xl relative overflow-hidden">
                    {/* Welcome Step */}
                    {step === 'welcome' && (
                        <div className="text-center space-y-6">
                            <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-accent-500 rounded-3xl flex items-center justify-center mx-auto animate-pulse">
                                <span className="text-4xl">🤖</span>
                            </div>
                            <h1 className="text-3xl font-bold text-white">Meet Pi</h1>
                            <p className="text-gray-400 text-lg leading-relaxed">
                                I'm your universal agent harness. I can code, browse, and control your system to help you get things done.
                            </p>
                            <button
                                onClick={handleNext}
                                className="w-full py-4 bg-primary-600 hover:bg-primary-500 text-white font-bold rounded-xl transition-all"
                            >
                                Let's get started
                            </button>
                        </div>
                    )}

                    {/* Model Selection Step */}
                    {step === 'model' && (
                        <div className="space-y-6">
                            <h2 className="text-2xl font-bold text-white">Select Pi's Brain</h2>
                            <p className="text-sm text-gray-400">Choose where Pi's intelligence comes from.</p>

                            <div className="space-y-3">
                                <ProviderCard
                                    id="anthropic"
                                    name="Anthropic Claude"
                                    desc="Best for coding and logical reasoning."
                                    icon="🧠"
                                    selected={provider === 'anthropic'}
                                    onClick={() => setProvider('anthropic')}
                                />
                                <ProviderCard
                                    id="gemini"
                                    name="Google Gemini"
                                    desc="Advanced and high-speed intelligence."
                                    icon="✨"
                                    selected={provider === 'gemini'}
                                    onClick={() => setProvider('gemini')}
                                />
                                <ProviderCard
                                    id="local"
                                    name="Local Model"
                                    desc="100% private, runs on your own hardware."
                                    icon="🏠"
                                    selected={provider === 'local'}
                                    onClick={() => setProvider('local')}
                                />
                            </div>

                            <button
                                onClick={handleNext}
                                className="w-full py-4 bg-primary-600 hover:bg-primary-500 text-white font-bold rounded-xl transition-all mt-4"
                            >
                                Next Step
                            </button>
                        </div>
                    )}

                    {/* API Key / OAuth Step */}
                    {step === 'api-key' && (
                        <div className="space-y-6">
                            <h2 className="text-2xl font-bold text-white">Authentication</h2>
                            <p className="text-sm text-gray-400">
                                Connect Pi to your {provider === 'anthropic' ? 'Anthropic' : 'Google'} account.
                            </p>

                            <div className="space-y-4">
                                {/* OAuth Path */}
                                <div className="p-4 bg-primary-500/5 rounded-2xl border border-primary-500/20 space-y-4">
                                    <div className="flex items-center justify-between">
                                        <h3 className="text-sm font-bold text-primary-400">OAuth Method (Recommended)</h3>
                                        {provider === 'gemini' && (
                                            <button
                                                onClick={() => setShowAdvancedAuth(!showAdvancedAuth)}
                                                className="text-[10px] text-primary-500 hover:underline"
                                            >
                                                {showAdvancedAuth ? 'Hide Advanced' : 'Custom Credentials?'}
                                            </button>
                                        )}
                                    </div>

                                    {!showAdvancedAuth && provider === 'gemini' && (
                                        <p className="text-xs text-gray-500">
                                            Seamless "Antigravity" login. No setup required.
                                        </p>
                                    )}

                                    {showAdvancedAuth && (
                                        <div className="space-y-2 animate-in fade-in slide-in-from-top-2 duration-300">
                                            <p className="text-[10px] text-gray-500 mb-2">
                                                Use your own Google Cloud Client ID. Redirect URI must be <code>http://localhost:5678/callback</code>.
                                            </p>
                                            <input
                                                type="text"
                                                placeholder="Client ID"
                                                id="oauth-client-id"
                                                className="w-full p-3 bg-black/40 border border-white/5 rounded-lg text-sm text-white focus:border-primary-500 outline-none"
                                            />
                                            <input
                                                type="password"
                                                placeholder="Client Secret"
                                                id="oauth-client-secret"
                                                className="w-full p-3 bg-black/40 border border-white/5 rounded-lg text-sm text-white focus:border-primary-500 outline-none"
                                            />
                                        </div>
                                    )}

                                    <button
                                        onClick={async () => {
                                            const clientId = showAdvancedAuth ? (document.getElementById('oauth-client-id') as HTMLInputElement).value : null;
                                            const clientSecret = showAdvancedAuth ? (document.getElementById('oauth-client-secret') as HTMLInputElement).value : null;

                                            if (showAdvancedAuth && (!clientId || !clientSecret)) {
                                                alert('Please enter both Client ID and Secret for manual setup.');
                                                return;
                                            }

                                            setIsLoading(true);
                                            try {
                                                const code = await invoke<string>('start_oauth', {
                                                    provider,
                                                    clientId: clientId || undefined
                                                });

                                                await invoke('exchange_oauth_code', {
                                                    provider,
                                                    code,
                                                    clientId: clientId || undefined,
                                                    clientSecret: clientSecret || undefined,
                                                    redirectUri: showAdvancedAuth ? "http://localhost:5678/callback" : undefined
                                                });
                                                setStep('skills');
                                            } catch (e) {
                                                alert('Authentication failed: ' + e);
                                            } finally {
                                                setIsLoading(false);
                                            }
                                        }}
                                        className="w-full py-3 bg-white text-black font-bold rounded-xl flex items-center justify-center gap-3 hover:bg-gray-200 transition-all"
                                    >
                                        <span>{provider === 'gemini' ? '🇬' : '🅰️'}</span>
                                        Sign in with {provider === 'gemini' ? 'Google' : 'Anthropic'}
                                    </button>
                                </div>

                                <div className="flex items-center gap-4 text-gray-600">
                                    <div className="flex-1 h-px bg-white/10" />
                                    <span className="text-xs uppercase font-bold">or use API key</span>
                                    <div className="flex-1 h-px bg-white/10" />
                                </div>

                                {/* Manual Key Path */}
                                <input
                                    type="password"
                                    value={apiKey}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    placeholder={`Enter ${provider} API key...`}
                                    className="w-full p-4 bg-black/40 border border-white/10 rounded-xl text-white focus:border-primary-500 outline-none"
                                />
                            </div>

                            <div className="flex gap-3 mt-4">
                                <button onClick={() => setStep('model')} className="flex-1 py-4 bg-gray-800 text-white rounded-xl">Back</button>
                                <button
                                    onClick={handleNext}
                                    disabled={isLoading || (!apiKey && step === 'api-key')}
                                    className="flex-[2] py-4 bg-primary-600 hover:bg-primary-500 text-white font-bold rounded-xl transition-all disabled:opacity-50"
                                >
                                    {isLoading ? 'Saving...' : 'Connect Brain'}
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Skills Selection Step */}
                    {step === 'skills' && (
                        <div className="space-y-6">
                            <h2 className="text-2xl font-bold text-white">Enable Core Skills</h2>
                            <p className="text-sm text-gray-400">What should Pi be allowed to do?</p>

                            <div className="space-y-3">
                                <SkillToggle
                                    id="shell"
                                    name="Shell Access"
                                    desc="Execute terminal commands and run scripts."
                                    enabled={skills.shell}
                                    onToggle={() => setSkills({ ...skills, shell: !skills.shell })}
                                />
                                <SkillToggle
                                    id="browser"
                                    name="Browser Control"
                                    desc="Search the web and interact with websites."
                                    enabled={skills.browser}
                                    onToggle={() => setSkills({ ...skills, browser: !skills.browser })}
                                />
                                <SkillToggle
                                    id="files"
                                    name="File System"
                                    desc="Read and write files within allowed paths."
                                    enabled={skills.files}
                                    onToggle={() => setSkills({ ...skills, files: !skills.files })}
                                />
                            </div>

                            <button
                                onClick={handleNext}
                                disabled={isLoading}
                                className="w-full py-4 bg-primary-600 hover:bg-primary-500 text-white font-bold rounded-xl transition-all mt-4"
                            >
                                {isLoading ? 'Configuring...' : 'Finish Setup'}
                            </button>
                        </div>
                    )}

                    {/* Hatching Step */}
                    {step === 'hatching' && (
                        <div className="text-center space-y-6">
                            <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-accent-500 rounded-3xl flex items-center justify-center mx-auto shadow-lg shadow-primary-500/20">
                                <span className="text-4xl text-white">🤖</span>
                            </div>
                            <h2 className="text-3xl font-bold text-white">Pi is Ready</h2>
                            <div className="text-gray-300 text-left bg-white/5 p-6 rounded-2xl border border-white/5 max-h-60 overflow-y-auto italic">
                                {formatMessage(hatchingMessage)}
                            </div>
                            <button
                                onClick={handleComplete}
                                className="w-full py-4 bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-500 hover:to-accent-500 text-white font-bold rounded-xl shadow-lg transition-all hover:scale-[1.02] active:scale-[0.98]"
                            >
                                Begin Journey
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function ProviderCard({ name, desc, icon, selected, onClick }: any) {
    return (
        <div
            onClick={onClick}
            className={`p-4 rounded-2xl border cursor-pointer transition-all ${selected ? 'bg-primary-500/10 border-primary-500 shadow-lg shadow-primary-500/5' : 'bg-white/5 border-white/5 hover:border-white/20'
                }`}
        >
            <div className="flex items-center gap-4">
                <div className="text-2xl">{icon}</div>
                <div className="flex-1 text-left">
                    <div className="font-bold text-white text-sm">{name}</div>
                    <div className="text-xs text-gray-400">{desc}</div>
                </div>
                {selected && <div className="text-primary-500">✓</div>}
            </div>
        </div>
    );
}

function SkillToggle({ name, desc, enabled, onToggle }: any) {
    return (
        <div
            onClick={onToggle}
            className="p-4 rounded-2xl bg-white/5 border border-white/5 flex items-center gap-4 cursor-pointer hover:border-white/20 transition-all"
        >
            <div className="flex-1 text-left">
                <div className="font-bold text-white text-sm">{name}</div>
                <div className="text-xs text-gray-400">{desc}</div>
            </div>
            <div className={`w-12 h-6 rounded-full transition-colors relative ${enabled ? 'bg-primary-600' : 'bg-gray-700'}`}>
                <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-all ${enabled ? 'right-1' : 'left-1'}`} />
            </div>
        </div>
    );
}

function formatMessage(text: string): JSX.Element[] {
    if (!text) return [<p key="empty" className="animate-pulse">Loading introduction...</p>];
    return text.split('\n').map((line, i) => {
        const formatted = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        return (
            <p
                key={i}
                className={`mb-2 ${line.startsWith('-') ? 'ml-4' : ''}`}
                dangerouslySetInnerHTML={{ __html: formatted || '&nbsp;' }}
            />
        );
    });
}
