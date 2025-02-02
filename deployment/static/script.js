document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("video");
    const startButton = document.getElementById("startDetection");
    const resultsDiv = document.getElementById("results");

    // Access camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => video.srcObject = stream)
        .catch(err => console.error("Camera error:", err));

    // Capture frame and send it to Flask
    startButton.addEventListener("click", async () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob(blob => {
            const formData = new FormData();
            formData.append("video_frame", blob, "frame.jpg");

            fetch("/detect", { method: "POST", body: formData })
                .then(response => response.json())
                .then(data => {
                    resultsDiv.innerHTML = `
                        <p>🧠 Emotion: ${data.emotion}</p>
                        <p>👀 Gaze: ${data.gaze}</p>
                        <p>🎥 Head Movement: ${data.head_movement}</p>
                        <p>🛠 Objects: ${data.objects}</p>
                    `;
                });
        }, "image/jpeg");
    });
});
