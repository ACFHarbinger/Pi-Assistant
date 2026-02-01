describe("Hatching Experience", () => {
  beforeEach(() => {
    // Clear local storage to ensure hatching starts
    cy.clearLocalStorage();

    // Mock Tauri invoke
    cy.visitWithMock("/");
  });

  it("completes the hatching flow", () => {
    // Welcome Step
    cy.contains("Meet Pi").should("be.visible");
    cy.contains("Let's get started").click();

    // Model Selection Step
    cy.contains("Select Pi's Brain").should("be.visible");
    cy.contains("Google Antigravity").click(); // Default
    cy.contains("Next Step").click();

    // API Key Step
    cy.contains("Authentication").should("be.visible");
    // Enter a fake API key
    cy.get('input[type="password"]').type("fake-api-key");
    cy.contains("Connect Brain").click();

    // Skills Step
    cy.contains("Enable Core Skills").should("be.visible");
    cy.contains("Finish Setup").click();

    // Identity Step
    cy.contains("Name Your Agent").should("be.visible");
    cy.get('input[placeholder*="Agent Name"]').clear().type("Cypress Agent");
    cy.contains("Confirm Name").click();

    // Hatching Phase
    cy.contains("Interactive Hatching", { timeout: 10000 }).should(
      "be.visible",
    );

    // Test Chat
    cy.get('input[placeholder="Talk to your new friend..."]').type(
      "Hello!{enter}",
    );
    cy.contains("I'm ready!").should("be.visible");

    // Complete
    cy.contains("Begin Journey").click();

    // Verify completion
    cy.should(() => {
      expect(localStorage.getItem("pi-hatched")).to.eq("true");
    });
  });
});
