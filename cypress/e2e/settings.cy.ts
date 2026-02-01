describe("Settings", () => {
  beforeEach(() => {
    localStorage.setItem("pi-hatched", "true");
    cy.visitWithMock("/");
  });

  it("opens and interacts with settings", () => {
    // Open Settings
    cy.get('button[title="Settings"]').click();
    cy.contains("Settings").should("be.visible");

    // Check Tabs
    cy.contains("MCP Servers").click();
    cy.contains("Add Server").should("be.visible");

    cy.contains("Tools").click();
    cy.contains("Only shows tools explicitly toggled").should("be.visible");

    cy.contains("Models").click();
    cy.contains("Add Model").should("be.visible");

    // Close
    cy.contains("✕").click();
    cy.contains("Settings").should("not.exist");
  });
});
