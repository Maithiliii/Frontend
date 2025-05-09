document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");
    const resultsSection = document.getElementById("resultsSection");
    const loadingIndicator = document.getElementById("loading");

    window.processImage = function (organ) {
        const file = imageInput.files[0];
        if (!file) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        loadingIndicator.classList.remove("hidden");

        resultsSection.classList.add("hidden");

        fetch(`/process/${organ}`, {
            method: "POST",
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                loadingIndicator.classList.add("hidden");

                resultsSection.classList.remove("hidden");

                const originalImage = document.getElementById("originalImage");
                originalImage.src = data.original;
                originalImage.classList.remove("hidden");

                const denoisedImage = document.getElementById("denoisedImage");
                denoisedImage.src = data.denoised;
                denoisedImage.classList.remove("hidden");

                const predictionBox = document.getElementById("predictionBox");
                predictionBox.innerText = `Predicted: ${data.prediction}`;
                predictionBox.classList.remove("hidden");
            })
            .catch(error => {
                alert("Error processing image!");
                console.error(error);
            });
    };
});
