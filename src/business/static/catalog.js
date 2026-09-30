(() => {
  'use strict';
  const el = (id) => document.getElementById(`catalog-${id}`);
  let products = [], editing = null, busy = false;
  function status(message, error = false) {
    el('status').textContent = message;
    el('status').classList.toggle('error', error);
  }
  function setBusy(value) {
    busy = value;
    el('fields').disabled = value;
    el('add').disabled = value;
    el('refresh').disabled = value;
    el('list').querySelectorAll('button').forEach((button) => { button.disabled = value; });
  }
  async function request(url, options) {
    const response = await fetch(url, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error?.message || `请求失败（${response.status}）`);
    return body;
  }
  function render() {
    el('list').replaceChildren();
    for (const product of products) {
      const card = document.createElement('article');
      card.className = 'product-card';
      card.dataset.productId = product.id;
      const name = document.createElement('h3');
      name.textContent = product.name;
      const size = document.createElement('small');
      size.textContent = product.size || '未填写规格';
      const info = document.createElement('p');
      info.textContent = `剩余 ${product.stock} 瓶 · ${product.price_yuan} 元/瓶${product.stock === 0 ? ' · 缺货' : ''}`;
      if (product.stock === 0) info.className = 'sold-out';
      const edit = document.createElement('button');
      edit.className = 'secondary';
      edit.textContent = '编辑';
      edit.setAttribute('aria-label', `编辑${product.name}`);
      edit.onclick = () => open(product);
      card.append(name, size, info, edit);
      el('list').appendChild(card);
    }
  }
  function open(product = null) {
    if (busy) return;
    editing = product?.id || null;
    el('form').reset();
    el('form-title').textContent = product ? '编辑饮料' : '新增饮料';
    for (const [field, value] of Object.entries({
      name: product?.name || '', size: product?.size || '', stock: product?.stock ?? '',
      price: product?.price_yuan || '', tags: (product?.tags || []).join('，'),
      aliases: (product?.aliases || []).join('，'),
    })) el(field).value = value;
    el('form').hidden = false;
    el('name').focus();
  }
  async function refresh() {
    if (busy) return;
    setBusy(true);
    try {
      products = (await request('/api/retail/catalog')).products;
      render();
      status('商品资料已加载。修改后新导购会话生效，独立语音识别下次请求生效。');
    } catch (error) { status(`加载失败：${error.message}`, true); }
    finally { setBusy(false); }
  }
  const words = (value) => [...new Set(value.split(/[，,\n]/).map((s) => s.trim()).filter(Boolean))];
  el('form').onsubmit = async (event) => {
    event.preventDefault();
    if (busy || !el('form').reportValidity()) return;
    const values = {
      name: el('name').value.trim(), size: el('size').value.trim(),
      stock: Number(el('stock').value), price_yuan: el('price').value,
      tags: words(el('tags').value), aliases: words(el('aliases').value),
    };
    setBusy(true);
    status('正在保存…');
    try {
      const product = await request(`/api/retail/products${editing ? `/${encodeURIComponent(editing)}` : ''}`, {
        method: editing ? 'PATCH' : 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(values),
      });
      const index = products.findIndex((p) => p.id === product.id);
      if (index < 0) products.push(product); else products[index] = product;
      render();
      el('form').hidden = true;
      status('已保存并同步热词。新导购会话生效，当前会话继续使用原资料；独立语音识别下次请求生效。');
    } catch (error) { status(`保存失败：${error.message}`, true); }
    finally { setBusy(false); }
  };
  el('add').onclick = () => open();
  el('cancel').onclick = () => { el('form').hidden = true; };
  el('refresh').onclick = refresh;
  refresh();
})();
