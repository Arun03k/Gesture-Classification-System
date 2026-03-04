// helper_methods.js — shared utilities for gesture-controlled slideshow

const uid = (function () {
    let i = 0;
    return function () { return "elem-" + (++i); };
})();

// Track rotation angle per element
const rotationAngles = {};

// Rotate all .rotatable elements by +90 deg
const rotateClockwise = function (slide) {
    const elems = Array.from(slide.getElementsByClassName("rotatable"));
    elems.forEach(function (elem) {
        if (!elem.id) elem.id = uid();
        rotationAngles[elem.id] = (rotationAngles[elem.id] || 0) + 90;
        elem.style.transform = "rotate(" + rotationAngles[elem.id] + "deg)";
    });
};

// Rotate all .rotatable elements by -90 deg
const rotateCounterClockwise = function (slide) {
    const elems = Array.from(slide.getElementsByClassName("rotatable"));
    elems.forEach(function (elem) {
        if (!elem.id) elem.id = uid();
        rotationAngles[elem.id] = (rotationAngles[elem.id] || 0) - 90;
        elem.style.transform = "rotate(" + rotationAngles[elem.id] + "deg)";
    });
};

// Show a temporary gesture overlay label on the slide
const showGestureOverlay = function (label) {
    let overlay = document.getElementById("gesture-overlay");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.id = "gesture-overlay";
        overlay.style.cssText = [
            "position:fixed", "top:20px", "right:20px",
            "background:rgba(0,0,0,0.65)", "color:#fff",
            "font-size:1.4rem", "padding:8px 18px",
            "border-radius:8px", "z-index:9999",
            "pointer-events:none", "transition:opacity 0.4s"
        ].join(";");
        document.body.appendChild(overlay);
    }
    overlay.textContent = "\uD83E\uDD32 " + label;
    overlay.style.opacity = "1";
    clearTimeout(overlay._timeout);
    overlay._timeout = setTimeout(() => { overlay.style.opacity = "0"; }, 1500);
};
