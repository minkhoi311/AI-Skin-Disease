    // Xem trước ảnh khi chọn file
    document.getElementById("file").addEventListener("change", function(event) {
        const preview = document.getElementById("preview");
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = "block";
            };
            reader.readAsDataURL(file);
        } else {
            preview.src = "";
            preview.style.display = "none";
        }
    });
    if (performance.navigation.type === performance.navigation.TYPE_RELOAD) {
        window.location.href = window.location.pathname;
    };

// phần loading
document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    const btn = document.querySelector(".btn-custom");

    form.addEventListener("submit", function() {
        btn.disabled = true;
        btn.innerHTML = "⏳ Đang phân tích...";
        document.getElementById("loading").style.display = "block";
    });
});

