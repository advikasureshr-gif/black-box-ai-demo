// =========================================================
// HELPERS
// =========================================================

function getTaskFromForm() {
    return {
        cpu_required: Number(
            document.getElementById("cpu").value
        ),

        memory_required: Number(
            document.getElementById("memory").value
        ),

        priority: Number(
            document.getElementById("priority").value
        ),

        deadline: Number(
            document.getElementById("deadline").value
        ),

        queue_length: Number(
            document.getElementById("queue").value
        )
    };
}


// =========================================================
// FORMAT LOREsa RULE
// =========================================================

function formatRule(rule) {

    if (!rule || !rule.premises) {
        return "No explanation available.";
    }

    const premises = rule.premises
        .map(premise => {
            return `${premise.attr} ${premise.op} ${premise.val}`;
        })
        .join("<br>AND ");

    const consequence = rule.consequence;

    return `
        ${premises}
        <br>
        <strong>
            → ${consequence.val}
        </strong>
    `;
}


// =========================================================
// FORMAT COUNTERFACTUAL
// =========================================================

function formatCounterfactual(counterfactual) {

    const premises = counterfactual.premises
        .map(premise => {
            return `${premise.attr} ${premise.op} ${premise.val}`;
        })
        .join(" AND ");

    const consequence = counterfactual.consequence;

    return `
        <div class="counterfactual">
            ${premises}
            <br>
            → ${consequence.val}
        </div>
    `;
}


// =========================================================
// SCHEDULE TASK
// =========================================================

async function scheduleTask() {

    const task = getTaskFromForm();

    const resultSection =
        document.getElementById("result-section");

    const explanationBox =
        document.getElementById("explanation-box");

    const blockedBox =
        document.getElementById("blocked-box");

    resultSection.classList.remove("hidden");

    blockedBox.classList.add("hidden");
    explanationBox.classList.remove("hidden");

    try {

        const response = await fetch(
            "/api/schedule",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(task)
            }
        );

        if (!response.ok) {
            throw new Error(
                "Server returned an error."
            );
        }

        const result = await response.json();

        // -------------------------------------------------
        // Scheduler decision
        // -------------------------------------------------

        document.getElementById(
            "assigned-vm"
        ).textContent = result.decision;


        // -------------------------------------------------
        // Probe score
        // -------------------------------------------------

        document.getElementById(
            "probe-score"
        ).textContent =
            Number(result.probe_score).toFixed(2);


        // -------------------------------------------------
        // Explanation allowed
        // -------------------------------------------------

        if (
            result.explanation_allowed &&
            result.explanation
        ) {

            explanationBox.classList.remove("hidden");
            blockedBox.classList.add("hidden");

            const explanation =
                result.explanation;

            document.getElementById(
                "rule"
            ).innerHTML =
                formatRule(explanation.rule);


            const counterfactualContainer =
                document.getElementById(
                    "counterfactuals"
                );

            counterfactualContainer.innerHTML = "";

            explanation.counterfactuals.forEach(
                counterfactual => {

                    counterfactualContainer.innerHTML +=
                        formatCounterfactual(
                            counterfactual
                        );
                }
            );


            document.getElementById(
                "fidelity"
            ).textContent =
                Number(
                    explanation.fidelity
                ).toFixed(2);

        } else {

            // -------------------------------------------------
            // Explanation blocked
            // -------------------------------------------------

            explanationBox.classList.add("hidden");
            blockedBox.classList.remove("hidden");

        }

    } catch (error) {

        console.error(error);

        alert(
            "Could not connect to the scheduler backend."
        );
    }
}


// =========================================================
// RUN BOUNDARY PROBE
// =========================================================

async function runProbe() {

    const probeResults =
        document.getElementById(
            "probe-results"
        );

    const rows =
        document.getElementById(
            "probe-rows"
        );

    const warning =
        document.getElementById(
            "probe-warning"
        );

    probeResults.classList.remove("hidden");

    rows.innerHTML = "";

    warning.classList.add("hidden");

    try {

        const response = await fetch(
            "/api/probe",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    start: 6,
                    end: 13
                })
            }
        );

        if (!response.ok) {
            throw new Error(
                "Server returned an error."
            );
        }

        const data =
            await response.json();


        // -------------------------------------------------
        // Render query sequence
        // -------------------------------------------------

        data.results.forEach(
            (result, index) => {

                const row =
                    document.createElement("div");

                row.className =
                    "probe-row";


                // Detect a decision transition
                // compared with previous query.

                let boundary = false;

                if (index > 0) {

                    const previous =
                        data.results[index - 1];

                    if (
                        previous.decision !==
                        result.decision
                    ) {
                        boundary = true;
                    }
                }


                if (boundary) {
                    row.classList.add(
                        "boundary-row"
                    );
                }


                row.innerHTML = `
                    <div class="probe-memory">
                        memory=${result.memory}
                    </div>

                    <div class="probe-vm">
                        → ${result.decision}
                    </div>

                    <div class="probe-score">
                        ${Number(
                            result.probe_score
                        ).toFixed(2)}
                    </div>
                `;


                rows.appendChild(row);
            }
        );


        // -------------------------------------------------
        // Determine whether probe was detected
        // -------------------------------------------------

        const detected =
            data.results.some(
                result => result.is_probe
            );


        if (detected) {

            warning.classList.remove(
                "hidden"
            );

        }

    } catch (error) {

        console.error(error);

        alert(
            "Could not connect to the scheduler backend."
        );
    }
}