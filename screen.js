const { chromium } = require('playwright');
(async () => {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    await page.setViewportSize({ width: 390, height: 844 });
    try {
        await page.goto('http://127.0.0.1:5175', { waitUntil: 'load', timeout: 10000 });
        // Wait for React to render (since it's an SPA without SSR)
        await new Promise(r => setTimeout(r, 2000));
        await page.screenshot({ path: '/Users/nithish/Desktop/MLSmartLift/screen.png' });
        console.log("Screenshot saved to /Users/nithish/Desktop/MLSmartLift/screen.png");
    } catch (e) {
        console.log("Error:", e.message);
    }
    await browser.close();
})();
