// Run against a local business service using an isolated RETAIL_CATALOG_FILE.
// This test creates and edits demo products. Do not point it at shared inventory.
const assert = require('node:assert/strict');
const {chromium} = require('playwright-core');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.APP_URL || 'http://127.0.0.1:18743');
    await page.waitForFunction(() => document.querySelectorAll('.product-card').length >= 4);
    assert.match(await page.locator('[data-product-id="cola"]').textContent(), /缺货/);
    await page.getByRole('button', {name: '新增饮料', exact: true}).click();
    const name = `浏览器饮料${Date.now()}`;
    await page.locator('#catalog-name').fill(name);
    await page.locator('#catalog-size').fill('480毫升');
    await page.locator('#catalog-stock').fill('2');
    await page.locator('#catalog-price').fill('4');
    await page.locator('#catalog-aliases').fill('测试简称，测试简称');
    await page.getByRole('button', {name: '保存饮料', exact: true}).click();
    await page.waitForFunction(() => document.getElementById('catalog-form').hidden);
    let card = page.locator('.product-card').filter({has: page.getByRole('heading', {name, exact: true})});
    assert.match(await card.textContent(), /剩余 2 瓶 · 4.00 元\/瓶/);
    assert.match(await page.locator('#catalog-status').textContent(), /新导购会话生效.*当前会话继续使用原资料/);
    await page.reload();
    await card.waitFor();
    assert.match(await card.textContent(), /剩余 2 瓶 · 4.00 元\/瓶/);

    await card.getByRole('button', {name: `编辑${name}`, exact: true}).click();
    assert.equal(await page.locator('#catalog-aliases').inputValue(), '测试简称');
    await page.locator('#catalog-stock').fill('0');
    await page.locator('#catalog-price').fill('3.50');
    await page.getByRole('button', {name: '保存饮料', exact: true}).click();
    await page.waitForFunction(() => document.getElementById('catalog-form').hidden);
    assert.match(await card.textContent(), /剩余 0 瓶 · 3.50 元\/瓶 · 缺货/);

    await page.getByRole('button', {name: '新增饮料', exact: true}).click();
    await page.locator('#catalog-name').fill(name);
    await page.locator('#catalog-size').fill('480毫升');
    await page.locator('#catalog-stock').fill('1');
    await page.locator('#catalog-price').fill('5');
    await page.getByRole('button', {name: '保存饮料', exact: true}).click();
    await page.waitForFunction(() => document.getElementById('catalog-status').textContent.includes('已存在'));
    assert.equal(await page.locator('#catalog-name').inputValue(), name);
    assert.equal(await page.locator('#catalog-form').isVisible(), true);

    await page.locator('#catalog-name').fill(`${name}新`);
    await page.route('**/api/retail/products', (route) => route.fulfill({status: 503,
      json: {error: {message: '商品资料保存失败，请稍后重试。'}},
    }), {times: 1});
    await page.getByRole('button', {name: '保存饮料', exact: true}).click();
    await page.waitForFunction(() => document.getElementById('catalog-status').textContent.includes('请稍后重试'));
    assert.equal(await page.locator('#catalog-name').inputValue(), `${name}新`);
    assert.equal(await page.getByRole('button', {name: '保存饮料', exact: true}).isEnabled(), true);
    await page.getByRole('button', {name: '取消', exact: true}).click();

    await page.route('**/api/retail/catalog', (route) => route.fulfill({status: 503,
      json: {error: {message: '商品资料暂时不可用。'}},
    }), {times: 1});
    await page.getByRole('button', {name: '刷新列表', exact: true}).click();
    await page.waitForFunction(() => document.getElementById('catalog-status').textContent.includes('加载失败'));
    await page.getByRole('button', {name: '刷新列表', exact: true}).click();
    await page.waitForFunction(() => document.getElementById('catalog-status').textContent.includes('已加载'));
    assert.deepEqual(errors, []);
    console.log('Catalog browser checks passed: display, add, edit, sold out, persistence, aliases, duplicate, save failure, reload recovery.');
  } finally { await browser.close(); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
