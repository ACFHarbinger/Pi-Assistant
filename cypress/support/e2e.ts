// Import commands.js using ES2015 syntax:
import "./commands";

Cypress.on("uncaught:exception", (err, runnable) => {
  // returning false here prevents Cypress from failing the test
  // because of Tauri internal errors in test environment
  return false;
});
