document.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById("imageInput");

    window.processImage = function (organ) {
        const file = imageInput.files[0];
        if (!file) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", file);

        document.getElementById("loading").classList.remove("hidden");

        fetch(`/process/${organ}`, {
            method: "POST",
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById("loading").classList.add("hidden");
                document.getElementById("originalImage").src = data.original;
                document.getElementById("originalImage").classList.remove("hidden");
                document.getElementById("denoisedImage").src = data.denoised;
                document.getElementById("denoisedImage").classList.remove("hidden");
            })
            .catch(error => {
                alert("Error processing image!");
                console.error(error);
            });
    }
});
