import { test, expect } from '@playwright/test';

test('overview, source blocks, accessible concepts, and real source', async ({page}) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.getByRole('heading', {name: /Two ways/})).toBeVisible();
  await expect(page.locator('.notice')).toHaveCount(0);
  await page.getByRole('button', {name: /Explore the Code/}).click();
  await page.getByRole('button', {name: 'Architecture', exact: false}).click();
  await expect(page.locator('.code-window')).toContainText('class VAE');
  await page.getByRole('button', {name:'GAN', exact:true}).click();
  await expect(page.locator('.code-window')).toContainText('ConvTranspose2d');
  await page.getByRole('button', {name:'Training step', exact:false}).click();
  await expect(page.locator('.code-window')).toContainText('fake.detach()');
  const detach = page.getByRole('button', {name:'detach ?'});
  await detach.focus();
  await expect(page.getByRole('tooltip').filter({hasText:'Remove a tensor'})).toBeVisible();
  await detach.click();
  await expect(detach).toHaveAttribute('aria-expanded','true');
  for (const label of ['Data pipeline','Learning objective','Sampling','Evaluation']) {
    await page.locator('.block-menu').getByRole('button', {name: new RegExp(label)}).click();
    await expect(page.locator('.code-window code')).not.toBeEmpty();
  }
  expect(errors).toEqual([]);
});

test('replay, seed selection, synchronization, reconstruction, zoom, download', async ({page}) => {
  await page.goto('/');
  await page.getByRole('button',{name:/Training Replay/}).click();
  await expect(page.locator('.epoch-pill').first()).toHaveText('EPOCH 0');
  const slider = page.getByRole('slider', {name:'Replay position'});
  await slider.fill('10');
  await expect(page.locator('.epoch-pill').first()).toHaveText('EPOCH 10');
  await expect(page.locator('.sample-image img').first()).toHaveAttribute('src', /epoch-010/);
  await page.getByLabel('Show test originals and reconstructions').check();
  await expect(page.locator('.sample-image img').first()).toHaveAttribute('src', /reconstruction-010/);
  await page.getByRole('button', {name:'Enlarge VAE samples'}).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('link',{name:'Download PNG'}).click();
  expect((await download).suggestedFilename()).toBe('sample-grid.png');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await page.getByLabel('Run seed', {exact:true}).selectOption('43');
  await expect(page.locator('.sample-image img').first()).toHaveAttribute('src', /seed-43/);
  await page.getByRole('button', {name:'Play replay'}).click();
  await expect(page.locator('.epoch-pill').first()).toHaveText('EPOCH 11', {timeout:3000});
  await page.getByRole('button', {name:'Pause replay'}).click();
  await page.getByLabel('Synchronize by', {exact:true}).selectOption('time');
  await expect(slider).toHaveAttribute('step', '0.1');
  await slider.fill('60');
  await expect(page.locator('.epoch-pill').first()).not.toHaveText('EPOCH 0');
  await expect(page.locator('canvas')).toHaveCount(5);
  await page.getByRole('button',{name:'Inspect run configuration'}).first().click();
  await expect(page.getByRole('dialog')).toContainText('source_sha256');
});

test('benchmark, all runs, metric selector and aggregate JSON', async ({page}) => {
  await page.goto('/');
  await page.getByRole('button',{name:/Benchmark/}).click();
  await expect(page.locator('tbody tr')).toHaveCount(6);
  await expect(page.locator('.n-label').first()).toHaveText('n=3');
  await page.getByLabel('Benchmark metric', {exact:true}).selectOption('entropy');
  await expect(page.locator('.metric-explanation')).toContainText('ln(10)');
  const download = page.waitForEvent('download');
  await page.getByRole('link',{name:'Download aggregate JSON'}).click();
  expect((await download).suggestedFilename()).toBe('benchmark-manifest.json');
  await page.getByText('Evaluator and protocol details', {exact:true}).click();
  await expect(page.locator('details')).toContainText('test_accuracy');
});

test('mobile layout and screenshot evidence', async ({page}) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  await expect(page.getByRole('heading', {name:/Two ways/})).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.screenshot({path:'test-results/mobile-overview.png', fullPage:true});
  await page.getByRole('button',{name:/Training Replay/}).click();
  await expect(page.locator('.sample-panel')).toHaveCount(2);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/');
  await expect(page.getByRole('heading', {name:/Two ways/})).toBeVisible();
  await page.screenshot({path:'test-results/desktop-overview.png', fullPage:true});
});

test('missing manifest is an explicit error', async ({page}) => {
  await page.route('**/data/manifest.json', route => route.fulfill({status:404, body:'missing'}));
  await page.goto('/');
  await expect(page.getByRole('alert')).toContainText('Recorded data unavailable');
});

test('missing run is unavailable, never fabricated', async ({page}) => {
  await page.route('**/vae-seed-42/run.json', route => route.fulfill({status:404, body:'missing'}));
  await page.goto('/');
  await expect(page.locator('.notice')).toContainText('5 of 6');
  await page.getByRole('button',{name:/Training Replay/}).click();
  await expect(page.getByRole('heading',{name:'VAE unavailable'})).toBeVisible();
});

test('missing image displays a message', async ({page}) => {
  await page.route('**/vae-seed-42/images/epoch-000.png', route => route.fulfill({status:404, body:'missing'}));
  await page.goto('/');
  await page.getByRole('button',{name:/Training Replay/}).click();
  await expect(page.locator('.image-error')).toContainText('Snapshot unavailable');
  await page.getByRole('slider', {name:'Replay position'}).fill('1');
  await expect(page.locator('.sample-image img').first()).toBeVisible();
  await expect(page.locator('.image-error')).toHaveCount(0);
});
