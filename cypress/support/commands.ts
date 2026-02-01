// ***********************************************
// This example commands.ts shows you how to
// create various custom commands and overwrite
// existing commands.
//
// For more comprehensive examples of custom
// commands please read more here:
// https://on.cypress.io/custom-commands
// ***********************************************

Cypress.Commands.add("visitWithMock", (url: string) => {
  return cy.visit(url, {
    onBeforeLoad(win: any) {
      const mockInvoke = (cmd: string, args: any) => {
        console.log("Mock invoke:", cmd, args);
        if (cmd === "save_current_model") return Promise.resolve();
        if (cmd === "save_api_key") return Promise.resolve();
        if (cmd === "toggle_tool") return Promise.resolve();
        if (cmd === "sidecar_request") {
          if (args?.method === "personality.update_name")
            return Promise.resolve();
          if (args?.method === "personality.hatch_chat") {
            return Promise.resolve({ text: "I'm ready!" });
          }
        }
        if (cmd === "get_mcp_config")
          return Promise.resolve({ mcpServers: {} });
        if (cmd === "get_tools_config")
          return Promise.resolve({ enabled_tools: {} });
        if (cmd === "list_local_models") return Promise.resolve([]);
        if (cmd === "get_telegram_config") return Promise.resolve({});
        if (cmd === "get_discord_config") return Promise.resolve({});
        return Promise.resolve();
      };

      const mockTransformCallback = (callback: any) => callback;

      win.__TAURI_INTERNALS__ = {
        invoke: mockInvoke,
        transformCallback: mockTransformCallback,
        metadata: {},
        plugins: {
          event: {
            unregisterListener: () => {},
          },
        },
      };
      win.__TAURI__ = {
        core: { invoke: mockInvoke },
        event: {
          listen: () => Promise.resolve(() => {}),
          emit: () => Promise.resolve(),
        },
      };
    },
  });
});

declare global {
  namespace Cypress {
    interface Chainable {
      visitWithMock(url: string): Chainable<AUTWindow>;
    }
  }
}
