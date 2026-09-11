import { Component, ElementRef, Input, OnChanges, OnDestroy, AfterViewInit, ViewChild } from '@angular/core';
import { Chart, registerables, Plugin } from 'chart.js';
import { Series } from './types';
Chart.register(...registerables);

@Component({selector: 'lab-chart', standalone: true, template: '<div class="chart-wrap"><canvas #canvas role="img" [attr.aria-label]="label"></canvas></div>@if(annotation){<p class="chart-annotation"><span class="annotation-dot"></span>{{annotation.label}}</p>}'})
export class ChartComponent implements AfterViewInit, OnChanges, OnDestroy {
  @ViewChild('canvas') canvas!: ElementRef<HTMLCanvasElement>;
  @Input() series: Series[] = [];
  @Input() label = 'Experiment metrics';
  @Input() cursor: number | null = null;
  @Input() xLabel = 'Epoch';
  @Input() bars = false;
  @Input() annotation: { x: number; label: string } | null = null;
  private chart?: Chart;
  ngAfterViewInit() { this.render(); }
  ngOnChanges() { if (this.canvas) this.render(); }
  ngOnDestroy() { this.chart?.destroy(); }
  private render() {
    this.chart?.destroy();
    const marker: Plugin = { id: 'epoch-marker', afterDraw: chart => {
      if (this.cursor === null || this.bars) return;
      const x = chart.scales['x'].getPixelForValue(this.cursor);
      if (x < chart.chartArea.left || x > chart.chartArea.right) return;
      const ctx = chart.ctx; ctx.save(); ctx.strokeStyle = '#24384688'; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, chart.chartArea.top); ctx.lineTo(x, chart.chartArea.bottom); ctx.stroke(); ctx.restore();
    }};
    const annotate: Plugin = { id: 'notable-epoch', afterDraw: chart => {
      if (!this.annotation || this.bars) return;
      const x = chart.scales['x'].getPixelForValue(this.annotation.x);
      if (x < chart.chartArea.left || x > chart.chartArea.right) return;
      const ctx = chart.ctx; ctx.save();
      ctx.strokeStyle = '#cc8b4b'; ctx.lineWidth = 1.5; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(x, chart.chartArea.top); ctx.lineTo(x, chart.chartArea.bottom); ctx.stroke();
      ctx.fillStyle = '#cc8b4b'; ctx.beginPath(); ctx.arc(x, chart.chartArea.top, 3.5, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }};
    this.chart = new Chart(this.canvas.nativeElement, {
      type: this.bars ? 'bar' : 'line',
      data: { datasets: this.series.map(s => ({label: s.label, data: s.points, borderColor: s.color, backgroundColor: this.bars ? s.color + 'bb' : s.color, borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, tension: 0.15})) },
      options: { responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 7, font: {size: 11} } }, tooltip: { enabled: true } },
        scales: { x: {type: 'linear', title: {display: true, text: this.xLabel}, ticks: {precision: 0}, grid: {display: false}}, y: {title: {display: false}, grid: {color: '#e8eced'}} } },
      plugins: [marker, annotate]
    });
  }
}
