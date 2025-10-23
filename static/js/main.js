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

// === Ẩn khung kết quả khi người dùng chọn ảnh mới hoặc nhập mô tả mới ===
const inputs = [document.getElementById("file"), document.getElementById("desc")];
inputs.forEach(input => {
    input.addEventListener("input", function () {
        const resultBox = document.querySelector(".result-box");
        if (resultBox) resultBox.style.display = "none";
    });
});

// === Thêm hiệu ứng nhẹ cho ảnh xem trước ===
const preview = document.getElementById("preview");
preview.addEventListener("mouseenter", () => {
    preview.style.transform = "scale(1.03)";
    preview.style.boxShadow = "0 0 10px rgba(0, 125, 125, 0.3)";
});
preview.addEventListener("mouseleave", () => {
    preview.style.transform = "scale(1)";
    preview.style.boxShadow = "none";
});

// thêm fade in out
window.addEventListener("DOMContentLoaded", () => {
    const resultBox = document.querySelector(".result-box");
    if (resultBox) {
        resultBox.style.opacity = 0;
        resultBox.style.transition = "opacity 0.6s ease-in-out";
        setTimeout(() => (resultBox.style.opacity = 1), 100);
    }
});


// Hiệu ứng thanh tiến trình xuất hiện mượt
window.addEventListener("DOMContentLoaded", () => {
    const bar = document.querySelector(".progress-bar");
    if (bar) {
        const target = parseFloat(bar.getAttribute("aria-valuenow"));
        bar.style.width = "0%";
        let current = 0;
        const timer = setInterval(() => {
            if (current >= target) {
                clearInterval(timer);
            } else {
                current += 1;
                bar.style.width = current + "%";
                bar.textContent = current + "%";
            }
        }, 15);
    }
});