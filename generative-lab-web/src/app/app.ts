import { Component, OnInit, OnDestroy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChartComponent } from './chart';
import { CodeBlock, Epoch, Manifest, Model, Run, Series } from './types';

@Component({selector: 'app-root', standalone: true, imports: [CommonModule, ChartComponent], templateUrl: './app.html'})
export class AppComponent implements OnInit, OnDestroy {
  readonly tabs = ['Overview', 'Explore the Code', 'Training Replay', 'Benchmark'];
  readonly models: Model[] = ['vae', 'gan'];
  readonly colors = {vae: '#167c80', gan: '#b25a32'};
  readonly seeds = [42, 43, 44];
  readonly sections = ['data', 'architecture', 'objective', 'training', 'sampling', 'evaluation'];
  readonly sectionLabels = ['Data pipeline', 'Architecture', 'Learning objective', 'Training step', 'Sampling', 'Evaluation'];
  readonly glossary: Record<string, string> = {
    'Latent space': 'A 64-dimensional representation. The VAE learns a distribution over it; the GAN maps random coordinates directly to images.',
    'KL divergence': 'A penalty encouraging the VAE posterior q(z|x) to resemble the standard Gaussian prior. It is not an image-quality score.',
    'Reparameterization': 'Write a random sample as a learned mean plus learned standard deviation times independent noise. Gradients can then reach the encoder.',
    'detach': 'Remove a tensor from the current gradient graph. In the discriminator update, fake.detach() prevents gradients reaching the generator.',
    'Discriminator': 'The GAN network that learns to distinguish training images from generated images. Its score is not an independently calibrated quality metric.',
    'Feature MMD': 'A distribution discrepancy in a frozen MNIST classifier’s feature space. Lower suggests closer distributions, but does not prove realism. This is not standard FID or KID.',
    'Entropy': 'Class entropy measures the spread of predicted digits. Its maximum is ln(10) ≈ 2.303. Noise or malformed images can still yield diverse predictions.',
    'Coverage': 'The number of predicted digit classes appearing at least once, from 0 to 10. Even one prediction counts; inspect the histogram for imbalance.',
    'Recorded replay': 'All displayed images and measurements were saved by Python. Playback advances recorded snapshots; it does not run a model or interpolate new images.'
  };
  tab = signal('Overview'); model = signal<Model>('vae'); section = signal(0); seed = signal(42);
  alignment = signal('epoch'); position = signal(0); speed = signal(1); playing = signal(false);
  manifest = signal<Manifest | null>(null); runs = signal<Run[]>([]); code = signal<Record<string, CodeBlock>>({});
  loading = signal(true); error = signal(''); warnings = signal<string[]>([]);
  term = signal<string | null>(null); zoom = signal<{url: string; title: string} | null>(null);
  configRun = signal<Run | null>(null); reconstruction = signal(false); metric = signal('feature_mmd');
  compare = signal(false); wipe = signal(50);
  scrubEpoch = signal<Record<string, number>>({});
  private timer?: ReturnType<typeof setInterval>; private lastFocus: HTMLElement | null = null;
  private restoringFromUrl = false;
  block = computed(() => {
    const key = this.sections[this.section()];
    return this.code()[['architecture', 'objective', 'training'].includes(key) ? `${this.model()}_${key}` : key];
  });
  selected = computed(() => this.models.map(model => this.runs().find(r => r.model === model && r.seed === this.seed())));
  maxPosition = computed(() => Math.max(0, ...this.selected().flatMap(r => r ? r.epochs.map(e => this.alignment() === 'epoch' ? e.epoch : e.training_seconds) : [])));
  completed = computed(() => this.runs().filter(r => r.status === 'complete').length);
  qualitySeries = computed(() => this.seriesFor(this.metric()));
  replaySeries = computed(() => this.seriesFor('feature_mmd'));
  lossSeries = computed(() => this.models.map(model => {
    const run = this.selected().find(r => r?.model === model);
    const keys = model === 'vae' ? ['total', 'reconstruction', 'kl'] : ['generator', 'discriminator'];
    return keys.map((key, i) => ({label: key, color: [this.colors[model], '#64748b', '#a29037'][i],
      points: run?.epochs.filter(e => e.losses !== null).map(e => ({x: this.alignment() === 'epoch' ? e.epoch : e.training_seconds, y: e.losses![key]})) ?? []}));
  }));
  discriminatorSeries = computed(() => {
    const run = this.selected().find(r => r?.model === 'gan');
    return ['d_real', 'd_fake'].map((key, i) => ({label: i === 0 ? 'D(real)' : 'D(fake), before G update', color: i === 0 ? '#64748b' : this.colors.gan,
      points: run?.epochs.filter(e => e.losses !== null).map(e => ({x: this.alignment() === 'epoch' ? e.epoch : e.training_seconds, y: e.losses![key]})) ?? []}));
  });
  digitSeries = computed(() => this.models.map(model => {
    const r = this.selected().find(r => r?.model === model); const e = r ? this.epochFor(r) : undefined;
    return { label: model.toUpperCase(), color: this.colors[model], points: e?.metrics.digit_counts.map((y, x) => ({x, y})) ?? [] };
  }));
  lossAnnotations = computed(() => this.models.map(model => {
    const run = this.selected().find(r => r?.model === model);
    const epochs = run?.epochs.filter(e => e.losses !== null) ?? [];
    if (epochs.length < 2) return null;
    const x = (e: Epoch) => this.alignment() === 'epoch' ? e.epoch : e.training_seconds;
    if (model === 'vae') {
      let best = epochs[1], drop = epochs[0].losses!['reconstruction'] - epochs[1].losses!['reconstruction'];
      for (let i = 2; i < epochs.length; i++) { const d = epochs[i - 1].losses!['reconstruction'] - epochs[i].losses!['reconstruction']; if (d > drop) { drop = d; best = epochs[i]; } }
      return drop > 0 ? { x: x(best), label: `Epoch ${best.epoch}: largest single-epoch drop in reconstruction loss (−${drop.toFixed(1)})` } : null;
    }
    let best = epochs[0], min = epochs[0].losses!['d_fake'];
    for (const e of epochs) if (e.losses!['d_fake'] < min) { min = e.losses!['d_fake']; best = e; }
    return { x: x(best), label: `Epoch ${best.epoch}: discriminator most confident on fakes (D(fake) = ${min.toFixed(3)})` };
  }));
  highlighted = computed(() => this.highlight(this.block()?.code ?? ''));
  async ngOnInit() {
    try {
      const [manifest, code] = await Promise.all([this.read<Manifest>('data/manifest.json'), this.read<Record<string, CodeBlock>>('data/code.json')]);
      if (manifest.schema_version !== 1) throw new Error('Unsupported dataset schema. Re-export with the current Python scripts.');
      this.manifest.set(manifest); this.code.set(code);
      const results = await Promise.allSettled(manifest.runs.map(r => this.read<Run>(`data/${r.path}`)));
      const loaded: Run[] = []; const warnings: string[] = [];
      results.forEach((result, i) => {
        if (result.status === 'fulfilled' && this.validRun(result.value)) loaded.push(result.value);
        else warnings.push(`Could not load ${manifest.runs[i].id}. Its results are unavailable.`);
      });
      loaded.filter(r => r.status !== 'complete').forEach(r => warnings.push(`${r.id} is ${r.status}; only recorded epochs are available.`));
      if (loaded.length < 6) warnings.push(`${loaded.length} of 6 expected runs are available. No missing measurements have been filled in.`);
      this.runs.set(loaded); this.warnings.set(warnings);
      this.restoreFromUrl();
    } catch (error) { this.error.set(error instanceof Error ? error.message : 'Unable to load recorded experiments.'); }
    finally { this.loading.set(false); }
  }
  ngOnDestroy() { this.pause(); }
  private restoreFromUrl() {
    const params = new URLSearchParams(location.search);
    this.restoringFromUrl = true;
    const tab = params.get('tab'); if (tab && this.tabs.includes(tab)) this.tab.set(tab);
    const model = params.get('model'); if (model === 'vae' || model === 'gan') this.model.set(model);
    const section = Number(params.get('section')); if (Number.isInteger(section) && section >= 0 && section < this.sections.length) this.section.set(section);
    const seed = Number(params.get('seed')); if (this.seeds.includes(seed)) this.seed.set(seed);
    const alignment = params.get('alignment'); if (alignment === 'epoch' || alignment === 'time') this.alignment.set(alignment);
    const position = Number(params.get('position')); if (Number.isFinite(position)) this.position.set(Math.max(0, Math.min(this.maxPosition(), position)));
    const metric = params.get('metric'); if (metric && ['feature_mmd', 'entropy', 'coverage', 'training_seconds'].includes(metric)) this.metric.set(metric);
    if (params.get('compare') === '1') this.compare.set(true);
    this.restoringFromUrl = false;
    this.syncUrl();
  }
  private syncUrl() {
    if (this.restoringFromUrl) return;
    const params = new URLSearchParams();
    params.set('tab', this.tab());
    if (this.tab() === 'Explore the Code') { params.set('model', this.model()); params.set('section', String(this.section())); }
    if (this.tab() === 'Training Replay' || this.tab() === 'Benchmark') {
      params.set('seed', String(this.seed())); params.set('alignment', this.alignment()); params.set('position', String(this.position()));
      if (this.tab() === 'Benchmark') params.set('metric', this.metric());
      if (this.tab() === 'Training Replay' && this.compare()) params.set('compare', '1');
    }
    history.replaceState(null, '', `${location.pathname}?${params}`);
  }
  private async read<T>(path: string): Promise<T> { const response = await fetch(path); if (!response.ok) throw new Error(`Missing artifact: ${path} (${response.status}). Export the experiments before opening the lab.`); return response.json(); }
  private validRun(r: Run) { return r.schema_version === 1 && Array.isArray(r.epochs) && r.epochs.length > 0 && r.epochs.every(e => Number.isFinite(e.metrics?.feature_mmd) && typeof e.image === 'string'); }
  navigate(tab: string) { this.tab.set(tab); this.pause(); this.term.set(null); this.syncUrl(); }
  selectModel(model: Model) { this.model.set(model); this.syncUrl(); }
  setSeed(event: Event) { this.pause(); this.seed.set(Number((event.target as HTMLSelectElement).value)); this.position.set(Math.min(this.position(), this.maxPosition())); this.syncUrl(); }
  setAlignment(event: Event) { this.pause(); this.alignment.set((event.target as HTMLSelectElement).value); this.position.set(0); this.syncUrl(); }
  setPosition(event: Event) { this.pause(); this.position.set(Number((event.target as HTMLInputElement).value)); this.syncUrl(); }
  togglePlayback() {
    if (this.playing()) { this.pause(); return; }
    if (!this.maxPosition()) return;
    if (this.position() >= this.maxPosition()) this.position.set(0);
    this.playing.set(true);
    this.timer = setInterval(() => {
      const delta = this.alignment() === 'epoch' ? 1 : this.maxPosition() / 20;
      this.position.set(Math.min(this.maxPosition(), this.position() + delta));
      if (this.position() >= this.maxPosition()) this.pause();
    }, 1100 / this.speed());
  }
  pause() { if (this.timer) clearInterval(this.timer); this.playing.set(false); }
  setSpeed(event: Event) { const wasPlaying = this.playing(); this.pause(); this.speed.set(Number((event.target as HTMLSelectElement).value)); if (wasPlaying) this.togglePlayback(); }
  step(direction: number) { this.pause(); this.position.set(Math.max(0, Math.min(this.maxPosition(), this.position() + direction * (this.alignment() === 'epoch' ? 1 : this.maxPosition() / 20)))); this.syncUrl(); }
  epochFor(run: Run): Epoch { return [...run.epochs].reverse().find(e => (this.alignment() === 'epoch' ? e.epoch : e.training_seconds) <= this.position() + 1e-8) ?? run.epochs[0]; }
  imageUrl(run: Run, epoch: Epoch) { return `data/${run.id}/${this.reconstruction() && run.model === 'vae' && epoch.reconstruction ? epoch.reconstruction : epoch.image}`; }
  seriesFor(key: string): Series[] {
    return this.models.map(model => ({label: model.toUpperCase(), color: this.colors[model], points: this.selected().find(r => r?.model === model)?.epochs.map(e => ({x: this.alignment() === 'epoch' ? e.epoch : e.training_seconds, y: key === 'training_seconds' ? e.training_seconds : e.metrics[key as 'feature_mmd' | 'entropy' | 'coverage']})) ?? []}));
  }
  setMetric(event: Event) { this.metric.set((event.target as HTMLSelectElement).value); this.syncUrl(); }
  toggleCompare() { this.compare.set(!this.compare()); this.wipe.set(50); this.syncUrl(); }
  setWipe(event: Event) { this.wipe.set(Number((event.target as HTMLInputElement).value)); }
  setSection(index: number) { this.section.set(index); this.syncUrl(); }
  scrubOverview(run: Run, event: MouseEvent) {
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const index = Math.round(ratio * (run.epochs.length - 1));
    this.scrubEpoch.update(m => ({...m, [run.id]: index}));
  }
  resetOverviewScrub(run: Run) { this.scrubEpoch.update(m => { const next = {...m}; delete next[run.id]; return next; }); }
  overviewEpoch(run: Run): Epoch { return run.epochs[this.scrubEpoch()[run.id] ?? run.epochs.length - 1]; }
  showTerm(term: string) { this.term.set(this.term() === term ? null : term); }
  openZoom(url: string, title: string) { this.lastFocus = document.activeElement as HTMLElement; this.zoom.set({url, title}); setTimeout(() => document.querySelector<HTMLButtonElement>('.modal-close')?.focus()); }
  openConfig(run: Run) { this.lastFocus = document.activeElement as HTMLElement; this.configRun.set(run); setTimeout(() => document.querySelector<HTMLButtonElement>('.modal-close')?.focus()); }
  closeModal() { this.zoom.set(null); this.configRun.set(null); this.lastFocus?.focus(); }
  modalKey(event: KeyboardEvent) {
    if (event.key === 'Escape') this.closeModal();
    if (event.key === 'Tab') { const buttons = (event.currentTarget as HTMLElement).querySelectorAll<HTMLElement>('button, a[href]'); if (!buttons.length) return; const first = buttons[0], last = buttons[buttons.length - 1]; if (event.shiftKey && document.activeElement === first) {event.preventDefault(); last.focus();} else if (!event.shiftKey && document.activeElement === last) {event.preventDefault(); first.focus();} }
  }
  imageLoaded(event: Event) { const img = event.target as HTMLImageElement; img.hidden = false; img.parentElement?.querySelectorAll('.image-error').forEach(el => el.remove()); }
  imageError(event: Event) { const img = event.target as HTMLImageElement; img.hidden = true; const message = document.createElement('span'); message.className = 'image-error'; message.textContent = 'Snapshot unavailable. No replacement image is shown.'; img.parentElement?.appendChild(message); }
  metricHelp() { return this.metric() === 'feature_mmd' ? 'Lower suggests closer real/generated feature distributions. Small negative estimates are possible; this is not standard FID or KID.' : this.metric() === 'entropy' ? 'Higher means more evenly distributed predicted digits, up to ln(10) ≈ 2.303. It does not establish visual quality.' : this.metric() === 'coverage' ? 'Number of predicted classes appearing at least once. A maximum of 10 can hide severe class imbalance.' : 'Cumulative optimizer-loop wall time. Evaluation and checkpoint serialization are outside this measurement.'; }
  format(value: number | null | undefined, digits = 3) { return value === null || value === undefined ? '—' : value.toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: digits}); }
  duration(seconds: number | undefined) { return seconds === undefined ? '—' : seconds < 60 ? `${seconds.toFixed(1)}s` : `${(seconds / 60).toFixed(1)} min`; }
  private highlight(code: string) { return code.split(/("[^"\n]*"|'[^'\n]*'|#[^\n]*|\b(?:def|class|return|if|else|for|in|with|import|from|as|None|True|False|not|and|or)\b|\b\d+(?:\.\d+)?\b)/g).map(part => { const safe = part.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); const cls = part.startsWith('#') ? 'comment' : /^['"]/.test(part) ? 'string' : /^(def|class|return|if|else|for|in|with|import|from|as|None|True|False|not|and|or)$/.test(part) ? 'keyword' : /^\d/.test(part) ? 'number' : ''; return cls ? `<span class="${cls}">${safe}</span>` : safe; }).join(''); }
}
