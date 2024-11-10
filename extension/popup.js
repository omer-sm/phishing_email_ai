document.getElementById("actionButton").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: () => {
                const emailText = document.getElementsByClassName("go")[0].textContent +
                " " + document.getElementsByClassName("hP")[0].textContent + 
                " " + document.getElementsByClassName("ii")[0].textContent
                
                return fetch("http://127.0.0.1:5000/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        text: emailText
                    })
                })
                .then(response => response.json())
            }
        }, res => {
            const elementContent = "This email was predicted to have a " + (Math.floor(res[0].result.prediction * 100)) + "% chance of being a phishing email."
            document.getElementById("response").textContent = elementContent || "Element not found!";
        });
    });
});
