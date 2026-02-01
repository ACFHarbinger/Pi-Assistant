describe("Dashboard", () => {
  beforeEach(() => {
    localStorage.setItem("pi-hatched", "true");
    cy.visitWithMock("/");
  });

  it("displays dashboard components", () => {
    cy.contains("Pi-Assistant", { timeout: 10000 }).should("be.visible");
    cy.get("header").should("be.visible");

    // Task Input
    cy.get('textarea[placeholder="What can I help you with today?"]').should(
      "be.visible",
    );

    // Status Card
    cy.contains("Agent Status").should("be.visible");
    cy.contains("Idle").should("be.visible");
  });

  it("allows entering a task", () => {
    cy.get('textarea[placeholder="What can I help you with today?"]').type(
      "New Task{enter}",
    );
    cy.get("textarea").should("have.value", "");
  });
});
