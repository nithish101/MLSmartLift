const { chromium } = require('playwright');
(async () => {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    try {
        const response = await page.goto('http://127.0.0.1:5174', { timeout: 10000, waitUntil: 'load' });
        console.log("STATUS:", response.status());
        console.log("HTML STARTING:", (await page.content()).substring(0, 500));
    } catch (e) {
        console.log("GOTO ERROR:", e.message);
    }
    await browser.close();
})();
