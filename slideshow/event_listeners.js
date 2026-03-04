// event_listeners.js — WebSocket gesture event handler

// Connect to Sanic WebSocket server
let socket;
let reconnectDelay = 2000;

function connectWS() {
    socket = new WebSocket("ws://127.0.0.1:8000/events");

    socket.onopen = function () {
        console.log("[WS] Connected to gesture server");
        reconnectDelay = 2000;
    };

    socket.onmessage = function (event) {
        const gesture = event.data.trim();
        console.log("[WS] Received gesture:", gesture);
        showGestureOverlay(gesture);

        const slide = Reveal.getCurrentSlide();

        switch (gesture) {
            case "sl":          // swipe left  → next slide
                Reveal.right();
                break;
            case "sr":          // swipe right → previous slide
                Reveal.left();
                break;
            case "su":          // swipe up    → sub-slide up / previous vertical
                Reveal.up();
                break;
            case "sd":          // swipe down  → sub-slide down / next vertical
                Reveal.down();
                break;
            case "r_cw":        // rotate clockwise → rotate image on slide
                rotateClockwise(slide);
                break;
            case "r_ccw":       // rotate counter-clockwise → rotate image back
                rotateCounterClockwise(slide);
                break;
            case "idle":
                // no action for idle
                break;
            default:
                console.debug("[WS] Unknown gesture:", gesture);
        }
    };

    socket.onerror = function (err) {
        console.warn("[WS] Error:", err);
    };

    socket.onclose = function () {
        console.warn("[WS] Disconnected. Reconnecting in", reconnectDelay, "ms ...");
        setTimeout(connectWS, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 1.5, 15000);
    };
}

// Auto-connect once the DOM is ready (before Reveal initializes)
document.addEventListener("DOMContentLoaded", connectWS);
