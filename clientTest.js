fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify({
        text: "qydlqcws-iacfym@issues.apache.org,xrh@spamassassin.apache.org [Bug 5780] URI processing turns uuencoded strings into http URI's which then causes FPs,\"http://issues.apache.org/SpamAssassin/show_bug.cgi?id=5780 wrzzpv@sidney.com changed: What    |Removed                     |Added ---------------------------------------------------------------------------- OtherBugsDependingO|                            |5813 nThis|                            | ------- You are receiving this mail because: ------- You are the assignee for the bug, or are watching the assignee."
    })
})
.then(response => response.json())
.then(data => {
    console.log("Prediction:", data.prediction);
})
.catch(error => {
    console.error("Error:", error);
});
