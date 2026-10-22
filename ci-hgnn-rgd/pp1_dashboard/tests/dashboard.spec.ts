import { test, expect } from '../frontend/node_modules/@playwright/test';
import path from 'node:path';

test('real artifacts, all pages and responsive navigation', async ({ page }, info) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Research overview', exact: true })).toBeVisible();
  await expect(page.getByText('Reading published artifacts').first()).toBeHidden({ timeout: 60_000 });
  await expect(page.getByTestId('mode-real')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('demo-banner')).toHaveCount(0);
  await expect(page.getByTestId('graph-canvas')).toHaveAttribute('data-node-count', /[1-9]/);
  await page.screenshot({ path: path.resolve(__dirname, `../reports/overview-${info.project.name}.png`), fullPage: true });
  const destinations = [ ['intents', 'Citation intent explorer'], ['topics', 'Scientific topic explorer'], ['graph', 'Heterogeneous knowledge graph'], ['gaps', 'Research gap explorer'], ['validation', 'Validation & progress'] ];
  for (const [id, title] of destinations) {
    if (info.project.name === 'mobile') await page.getByRole('button', { name: 'Open navigation', exact: true }).click();
    await page.getByTestId(`nav-${id}`).click();
    await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
    await expect(page.locator('main').getByRole('alert')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBeTruthy();
    if (id === 'intents') {
      await page.getByRole('button', { name: 'Inspect', exact: false }).first().click();
      await expect(page.getByTestId('citation-detail')).toBeVisible();
      await page.getByLabel('Citation intent', { exact: true }).selectOption('method');
      await expect(page.locator('tbody tr').first().getByText('method', { exact: true })).toBeVisible();
    }
    if (id === 'topics') {
      await page.locator('.topic-card').first().click();
      await expect(page.getByTestId('topic-detail')).toBeVisible();
    }
    if (id === 'graph') {
      await expect(page.getByTestId('graph-canvas')).toHaveAttribute('data-node-count', /[1-9]/);
      await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
      await page.getByRole('button', { name: 'Fit graph', exact: true }).click();
      const canvas = page.getByTestId('graph-canvas');
      expect(await canvas.locator('canvas').count()).toBeGreaterThan(0);
      const bounds = await canvas.boundingBox();
      if (bounds) await canvas.click({ position: { x: bounds.width / 2, y: bounds.height / 2 } });
      await expect(page.getByTestId('node-detail')).toBeVisible();
      await page.getByText('Visible nodes · keyboard navigation', { exact: true }).click();
      await page.locator('.visible-node-list button').first().click();
      await expect(page.getByTestId('node-detail')).toBeVisible();
      await page.screenshot({ path: path.resolve(__dirname, `../reports/graph-${info.project.name}.png`), fullPage: true });
      await page.getByRole('button', { name: 'Explore neighborhood', exact: true }).click();
      await page.getByLabel('Maximum visible nodes').selectOption('35');
      await expect.poll(async () => Number(await page.getByTestId('graph-canvas').getAttribute('data-node-count'))).toBeLessThanOrEqual(35);
    }
    if (id === 'gaps') await expect(page.getByText('Research-gap scoring will be available after Stage 08 implementation.', { exact: true })).toBeVisible();
  }
  expect(errors).toEqual([]);
});

test('explicit demo choice stays visibly separate across pages', async ({ page }, info) => {
  await page.goto('/#graph');
  await page.getByTestId('mode-demo').click();
  await expect(page.getByTestId('demo-banner')).toContainText('DEMO MODE — SYNTHETIC DATA — NOT RESEARCH RESULTS');
  await expect(page.getByTestId('graph-canvas')).toHaveAttribute('data-node-count', /[1-9]/);
  await expect(page.getByTestId('synthetic-label').first()).toContainText('Synthetic Demo Data — Not Research Findings');
  if (info.project.name === 'mobile') await page.getByRole('button', { name: 'Open navigation', exact: true }).click();
  await page.getByTestId('nav-validation').click();
  await expect(page.getByText('No verified numeric model-evaluation outputs are available in this mode.', { exact: false })).toBeVisible();
  await page.getByTestId('mode-real').click();
  await expect(page.getByTestId('demo-banner')).toHaveCount(0);
  await expect(page.getByTestId('synthetic-label')).toHaveCount(0);
});

test('missing artifacts remain real until an explicit demo action', async ({ page }) => {
  await page.route('**/api/**', (route) => route.fulfill({ json: { mode: 'real', status: 'unavailable', notice: 'No verified published artifact is installed.', graph_version: null, data: null } }));
  await page.goto('/');
  await expect(page.getByTestId('unavailable-state').first()).toBeVisible();
  await expect(page.getByTestId('mode-real')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('demo-banner')).toHaveCount(0);
});
