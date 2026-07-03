import {
  ChangeDetectorRef,
  Component,
  ElementRef,
  HostListener,
  OnDestroy,
  ViewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import embed, { Result } from 'vega-embed';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnDestroy {
  @ViewChild('chartContainer') chartContainer!: ElementRef<HTMLDivElement>;

  specText = '';
  errorMessage = '';
  chartReady = false;

  private vegaResult: Result | null = null;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private cdr: ChangeDetectorRef) {}

  @HostListener('window:resize')
  onWindowResize(): void {
    if (this.specText.trim()) this.renderChart();
  }

  onSpecChange(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => this.renderChart(), 400);
  }

  clearSpec(): void {
    this.specText = '';
    this.errorMessage = '';
    this.chartReady = false;
    this.clearChart();
  }

  private async renderChart(): Promise<void> {
    const raw = this.specText.trim();
    if (!raw) {
      this.clearChart();
      this.errorMessage = '';
      this.chartReady = false;
      this.cdr.detectChanges();
      return;
    }

    let spec: object;
    try {
      spec = JSON.parse(raw);
    } catch (e: any) {
      this.errorMessage = `JSON parse error: ${e.message}`;
      this.chartReady = false;
      this.cdr.detectChanges();
      return;
    }

    try {
      if (this.vegaResult) {
        this.vegaResult.finalize();
        this.vegaResult = null;
      }
      this.chartContainer.nativeElement.innerHTML = '';

      // Remove spec-level width/height — we compute them from the container
      delete (spec as any).width;
      delete (spec as any).height;

      const el = this.chartContainer.nativeElement;
      const width   = Math.max(el.clientWidth - 60, 200);
      // Available height: panel minus vertical padding (1.5rem×2≈48px) and title+legend (~80px)
      const available = Math.max(el.clientHeight - 48, 200);

      // For charts with a nominal y-axis (bar charts), set spec.height as a
      // band step so Vega-Lite scales every row to fit in the available height.
      // This is the only reliable way to make all rows visible without scroll.
      const encoding = (spec as any).encoding;
      const yNominal = encoding?.y?.type === 'nominal';
      if (yNominal) {
        const values: any[]  = (spec as any).data?.values ?? [];
        const yField         = encoding.y.field;
        const offsetField    = encoding?.yOffset?.field;
        const yDistinct      = new Set(values.map((r: any) => r[yField])).size;
        const offsetDistinct = offsetField
          ? new Set(values.map((r: any) => r[offsetField])).size
          : 1;
        const totalBands = Math.max(yDistinct * offsetDistinct, 1);
        // Divide available space among all bands; minimum 2px so all bars render
        const step = Math.max(Math.floor((available - 80) / totalBands), 2);
        (spec as any).height = { step };
      }

      const height = yNominal ? available : available;

      this.vegaResult = await embed(el, spec as any, {
        actions: { export: true, source: false, compiled: false, editor: false },
        renderer: 'svg',
        width,
        height,
      });

      this.errorMessage = '';
      this.chartReady = true;
    } catch (e: any) {
      this.errorMessage = `Render error: ${e.message}`;
      this.chartReady = false;
    }

    this.cdr.detectChanges();
  }

  private clearChart(): void {
    if (this.vegaResult) {
      this.vegaResult.finalize();
      this.vegaResult = null;
    }
    if (this.chartContainer?.nativeElement) {
      this.chartContainer.nativeElement.innerHTML = '';
    }
  }

  ngOnDestroy(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.clearChart();
  }
}
