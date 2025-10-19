const form = document.getElementById("backtest-form");
const statusText = document.getElementById("status-text");
const submitBtn = document.getElementById("submit-btn");
const metricsGrid = document.getElementById("metrics-grid");
const metricTemplate = document.getElementById("metric-item-template");
const tailTableBody = document.querySelector("#tail-table tbody");
const paginationInfo = document.getElementById("pagination-info");
const prevPageBtn = document.getElementById("prev-page");
const nextPageBtn = document.getElementById("next-page");

const equityCtx = document.getElementById("equity-chart").getContext("2d");
const returnsCtx = document.getElementById("returns-chart").getContext("2d");
const ohlcCtx = document.getElementById("ohlc-chart").getContext("2d");

let equityChart = null;
let returnsChart = null;
let ohlcChart = null;
let dailyRecords = [];
let currentPage = 1;
const PAGE_SIZE = 20;
let isEquityChartFocused = false;
let isOhlcChartFocused = false;
let equityCanvasDownHandler = null;
let ohlcCanvasDownHandler = null;
let documentClickHandler = null;
let documentKeydownHandler = null;

// Provide a lightweight date adapter using native Date so Chart.js time scales work without external libs
function overrideDateAdapter() {
  if (!window.Chart || !Chart._adapters || !Chart._adapters._date) {
    return;
  }

  const MILLISECONDS = {
    millisecond: 1,
    second: 1000,
    minute: 60000,
    hour: 3600000,
    day: 86400000,
    week: 604800000,
  };

  const FORMATS = {
    datetime: "yyyy-MM-dd HH:mm:ss",
    millisecond: "HH:mm:ss.SSS",
    second: "HH:mm:ss",
    minute: "HH:mm",
    hour: "HH:mm",
    day: "yyyy-MM-dd",
    week: "yyyy-'W'II",
    month: "yyyy-MM",
    quarter: "yyyy-'Q'q",
    year: "yyyy",
  };

  const clamp = (value) => (Number.isFinite(value) ? value : 0);

  const adapter = {
    _id: "native",
    formats: () => FORMATS,
    parse(value) {
      if (value === null || value === undefined) {
        return null;
      }
      if (value instanceof Date) {
        return value.getTime();
      }
      if (typeof value === "number") {
        return Number.isNaN(value) ? null : value;
      }
      if (typeof value === "string") {
        const parsed = Date.parse(value);
        return Number.isNaN(parsed) ? null : parsed;
      }
      if (value && typeof value.valueOf === "function") {
        const num = value.valueOf();
        return typeof num === "number" && !Number.isNaN(num) ? num : null;
      }
      return null;
    },
    format(time, formatStr) {
      if (time === null || time === undefined) {
        return "";
      }
      const date = new Date(time);
      if (Number.isNaN(date.getTime())) {
        return "";
      }
      const pad = (num, size = 2) => String(Math.abs(Math.trunc(num))).padStart(size, "0");
      const year = date.getFullYear();
      const month = pad(date.getMonth() + 1);
      const day = pad(date.getDate());
      const hours = pad(date.getHours());
      const minutes = pad(date.getMinutes());
      const seconds = pad(date.getSeconds());
      const millis = pad(date.getMilliseconds(), 3);

      switch (formatStr) {
        case "millisecond":
        case "second":
        case "minute":
        case "hour":
          return `${hours}:${minutes}${formatStr === "second" || formatStr === "millisecond" ? ":" + seconds : ""}${formatStr === "millisecond" ? "." + millis : ""}`;
        case "month":
          return `${year}-${month}`;
        case "quarter":
          return `${year}-Q${Math.floor(date.getMonth() / 3) + 1}`;
        case "year":
          return String(year);
        case "day":
        case "datetime":
        default:
          return `${year}-${month}-${day}`;
      }
    },
    add(time, amount, unit) {
      const date = new Date(time);
      if (Number.isNaN(date.getTime())) {
        return time;
      }
      switch (unit) {
        case "millisecond":
        case "second":
        case "minute":
        case "hour":
        case "day":
        case "week":
          date.setTime(date.getTime() + amount * MILLISECONDS[unit]);
          break;
        case "month":
          date.setMonth(date.getMonth() + amount);
          break;
        case "quarter":
          date.setMonth(date.getMonth() + amount * 3);
          break;
        case "year":
          date.setFullYear(date.getFullYear() + amount);
          break;
        default:
          date.setTime(date.getTime() + amount);
      }
      return date.getTime();
    },
    diff(max, min, unit) {
      const upper = max - min;
      switch (unit) {
        case "millisecond":
        case "second":
        case "minute":
        case "hour":
        case "day":
        case "week":
          return upper / MILLISECONDS[unit];
        case "month": {
          const end = new Date(max);
          const start = new Date(min);
          const years = end.getFullYear() - start.getFullYear();
          const months = end.getMonth() - start.getMonth();
          const days = end.getDate() - start.getDate();
          return years * 12 + months + days / 30;
        }
        case "quarter":
          return adapter.diff(max, min, "month") / 3;
        case "year":
          return adapter.diff(max, min, "month") / 12;
        default:
          return upper;
      }
    },
    startOf(time, unit) {
      const date = new Date(time);
      if (Number.isNaN(date.getTime())) {
        return time;
      }
      switch (unit) {
        case "second":
          date.setMilliseconds(0);
          break;
        case "minute":
          date.setSeconds(0, 0);
          break;
        case "hour":
          date.setMinutes(0, 0, 0);
          break;
        case "day":
          date.setHours(0, 0, 0, 0);
          break;
        case "week":
        case "isoWeek": {
          const day = (date.getDay() + 6) % 7; // Monday = 0
          date.setDate(date.getDate() - day);
          date.setHours(0, 0, 0, 0);
          break;
        }
        case "month":
          date.setDate(1);
          date.setHours(0, 0, 0, 0);
          break;
        case "quarter": {
          const month = date.getMonth();
          const quarterStart = month - (month % 3);
          date.setMonth(quarterStart, 1);
          date.setHours(0, 0, 0, 0);
          break;
        }
        case "year":
          date.setMonth(0, 1);
          date.setHours(0, 0, 0, 0);
          break;
        default:
          break;
      }
      return date.getTime();
    },
    endOf(time, unit) {
      if (unit === "isoWeek") {
        unit = "week";
      }
      const start = adapter.startOf(time, unit);
      if (unit === "millisecond") {
        return start;
      }
      const added = adapter.add(start, 1, unit === "year" ? "year" : unit);
      return added - 1;
    },
  };

  Chart._adapters._date.override(adapter);
}

overrideDateAdapter();

if (typeof Chart !== "undefined" && typeof Chart.register === "function" && typeof window.ChartZoom !== "undefined") {
  Chart.register(window.ChartZoom);
}

function setEquityChartFocus(enabled) {
  isEquityChartFocused = enabled;
  if (equityChart && equityChart.options?.plugins?.zoom) {
    const zoomOptions = equityChart.options.plugins.zoom;
    if (zoomOptions.zoom?.wheel) {
      zoomOptions.zoom.wheel.enabled = enabled;
    }
    if (zoomOptions.pan) {
      zoomOptions.pan.enabled = enabled;
    }
    equityChart.update("none");
  }
  const card = equityCtx?.canvas?.closest(".chart-card");
  if (card) {
    card.classList.toggle("chart-focused", enabled);
  }

  if (enabled) {
    if (!documentKeydownHandler) {
      documentKeydownHandler = (event) => {
        if (!isEquityChartFocused || !equityChart || typeof equityChart.pan !== "function") {
          return;
        }
        const step = event.shiftKey ? 120 : 60;
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          equityChart.pan({ x: step, y: 0 }, undefined, "none");
        } else if (event.key === "ArrowRight") {
          event.preventDefault();
          equityChart.pan({ x: -step, y: 0 }, undefined, "none");
        }
      };
      document.addEventListener("keydown", documentKeydownHandler);
    }
  } else if (documentKeydownHandler) {
    document.removeEventListener("keydown", documentKeydownHandler);
    documentKeydownHandler = null;
  }
}

function setOhlcChartFocus(enabled) {
  isOhlcChartFocused = enabled;
  if (ohlcChart && ohlcChart.options?.plugins?.zoom) {
    const zoomOptions = ohlcChart.options.plugins.zoom;
    if (zoomOptions.zoom?.wheel) {
      zoomOptions.zoom.wheel.enabled = enabled;
    }
    if (zoomOptions.pan) {
      zoomOptions.pan.enabled = enabled;
    }
    ohlcChart.update("none");
  }
  const card = ohlcCtx?.canvas?.closest(".chart-card");
  if (card) {
    card.classList.toggle("chart-focused", enabled);
  }
}

function attachEquityFocusHandlers() {
  const canvas = equityCtx?.canvas;
  if (!canvas) {
    return;
  }
  if (!equityCanvasDownHandler) {
    equityCanvasDownHandler = () => {
      setEquityChartFocus(true);
    };
    canvas.addEventListener("pointerdown", equityCanvasDownHandler);
  }
  if (!documentClickHandler) {
    documentClickHandler = (event) => {
      const canvasElement = equityCtx?.canvas;
      if (!canvasElement) {
        return;
      }
      if (!canvasElement.contains(event.target)) {
        setEquityChartFocus(false);
      }
      const ohlcCanvas = ohlcCtx?.canvas;
      if (ohlcCanvas && !ohlcCanvas.contains(event.target)) {
        setOhlcChartFocus(false);
      }
    };
    document.addEventListener("click", documentClickHandler);
  }
}

function attachOhlcFocusHandlers() {
  const canvas = ohlcCtx?.canvas;
  if (!canvas) {
    return;
  }
  if (!ohlcCanvasDownHandler) {
    ohlcCanvasDownHandler = () => {
      setOhlcChartFocus(true);
    };
    canvas.addEventListener("pointerdown", ohlcCanvasDownHandler);
  }
  if (!documentClickHandler) {
    documentClickHandler = (event) => {
      const equityCanvas = equityCtx?.canvas;
      if (equityCanvas && !equityCanvas.contains(event.target)) {
        setEquityChartFocus(false);
      }
      const ohlcCanvas = ohlcCtx?.canvas;
      if (ohlcCanvas && !ohlcCanvas.contains(event.target)) {
        setOhlcChartFocus(false);
      }
    };
    document.addEventListener("click", documentClickHandler);
  }
}

const metricConfig = {
  cumulative_return: { label: "累计收益", format: "percent", decimals: 2 },
  cagr: { label: "年化收益", format: "percent", decimals: 2 },
  volatility: { label: "年化波动", format: "percent", decimals: 2 },
  sharpe: { label: "夏普比率", format: "number", decimals: 2 },
  sortino: { label: "索提诺比率", format: "number", decimals: 2 },
  max_drawdown: { label: "最大回撤", format: "percent", decimals: 2 },
  calmar: { label: "卡玛比率", format: "number", decimals: 2 },
  hit_rate: { label: "胜率", format: "percent", decimals: 2 },
  avg_trade_return: { label: "单次交易平均收益", format: "percent", decimals: 2 },
  num_trades: { label: "交易次数", format: "integer" },
};

function formatMetric(key, value) {
  const config = metricConfig[key] || { label: key, format: "number", decimals: 2 };
  const label = config.label || key;

  if (value === undefined || value === null || Number.isNaN(value)) {
    return { label, value: "N/A" };
  }

  if (config.format === "integer") {
    return { label, value: Math.round(value).toString() };
  }

  if (typeof value !== "number") {
    return { label, value: value.toString() };
  }

  const decimals = config.decimals ?? 2;

  if (config.format === "percent") {
    return { label, value: `${(value * 100).toFixed(decimals)}%` };
  }

  return { label, value: value.toFixed(decimals) };
}

function renderMetrics(metrics) {
  metricsGrid.innerHTML = "";
  const entries = Object.entries(metrics);
  entries.forEach(([key, rawValue]) => {
    const { label, value } = formatMetric(key, rawValue);
    const clone = metricTemplate.content.firstElementChild.cloneNode(true);
    clone.querySelector(".metric-label").textContent = label;
    clone.querySelector(".metric-value").textContent = value;
    metricsGrid.appendChild(clone);
  });
}

function renderChart(ctx, previousChart, labels, dataset, options) {
  if (previousChart) {
    previousChart.destroy();
  }
  return new Chart(ctx, {
    type: options.type || "line",
    data: {
      labels,
      datasets: [dataset],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: options.scales,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index",
          intersect: false,
        },
      },
    },
  });
}

function updateCharts(equityCurve, returns) {
  const eqLabels = equityCurve.map((item) => item.date);
  const eqValues = equityCurve.map((item) => item.value);
  const eqMax = Math.max(...eqValues);
  const yMaxBound = eqMax === -Infinity ? undefined : eqMax * 1.1;

  equityChart = renderChart(
    equityCtx,
    equityChart,
    eqLabels,
    {
      label: "权益曲线",
      data: eqValues,
      borderColor: "#2563eb",
      backgroundColor: "rgba(37, 99, 235, 0.15)",
      fill: true,
      tension: 0.25,
    },
    {
      scales: {
        x: { display: true, ticks: { maxTicksLimit: 8 } },
        y: {
          display: true,
          suggestedMax: yMaxBound,
          ticks: { callback: (v) => v.toLocaleString() },
        },
      },
      plugins: {
        zoom: {
          zoom: {
            wheel: {
              enabled: isEquityChartFocused,
              modifierKey: null,
            },
            mode: "x",
            onZoom({ chart }) {
              adjustEquityYAxis(chart);
            },
          },
          pan: {
            enabled: isEquityChartFocused,
            mode: "x",
            onPan({ chart }) {
              adjustEquityYAxis(chart);
            },
          },
        },
      },
    },
  );
  attachEquityFocusHandlers();
  adjustEquityYAxis(equityChart);
  setEquityChartFocus(isEquityChartFocused);
  if (ohlcChart && ohlcChart.options?.plugins?.zoom) {
    const zoomOptions = ohlcChart.options.plugins.zoom;
    if (zoomOptions.zoom?.wheel) {
      zoomOptions.zoom.wheel.enabled = false;
    }
    if (zoomOptions.pan) {
      zoomOptions.pan.enabled = false;
    }
    ohlcChart.update("none");
  }

  const retLabels = returns.map((item) => item.date);
  const retValues = returns.map((item) => item.value * 100);
  const retMin = Math.min(...retValues);
  const retMax = Math.max(...retValues);
  const retRange = retMax - retMin;
  const retPadding =
    retRange === 0
      ? Math.max(Math.abs(retMax), 0.5)
      : Math.max(retRange * 0.15, 0.5);
  const retMinBound = Math.min(retMin - retPadding, -retPadding);
  const retMaxBound = Math.max(retMax + retPadding, retPadding);

  returnsChart = renderChart(
    returnsCtx,
    returnsChart,
    retLabels,
    {
      label: "日收益率",
      data: retValues,
      backgroundColor: retValues.map((v) => (v >= 0 ? "rgba(16, 185, 129, 0.6)" : "rgba(239, 68, 68, 0.6)")),
      borderWidth: 0,
    },
    {
      type: "bar",
      scales: {
        x: { display: true, ticks: { maxTicksLimit: 10 } },
        y: {
          display: true,
          suggestedMin: retMinBound,
          suggestedMax: retMaxBound,
          ticks: { callback: (v) => `${v.toFixed(1)}%` },
        },
      },
    },
  );
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return "-";
  }
  return numeric.toFixed(digits);
}

const formatPrice = (value) => formatNumber(value, 2);

function adjustEquityYAxis(chart) {
  if (!chart?.scales) {
    return;
  }
  const xScale = chart.scales.x;
  const yScale = chart.scales.y;
  const dataset = chart.data?.datasets?.[0];
  const labels = chart.data?.labels || [];

  if (!xScale || !yScale || !dataset || !Array.isArray(dataset.data) || dataset.data.length === 0) {
    return;
  }

  const toIndex = (value, fallback) => {
    if (value == null) {
      return fallback;
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    const idx = labels.indexOf(value);
    return idx === -1 ? fallback : idx;
  };

  let minIndex = toIndex(xScale.min, 0);
  let maxIndex = toIndex(xScale.max, dataset.data.length - 1);

  minIndex = Math.max(Math.floor(minIndex), 0);
  maxIndex = Math.min(Math.ceil(maxIndex), dataset.data.length - 1);
  if (maxIndex < minIndex) {
    [minIndex, maxIndex] = [minIndex, minIndex];
  }

  const visibleValues = dataset.data
    .slice(minIndex, maxIndex + 1)
    .filter((v) => Number.isFinite(v));

  if (!visibleValues.length) {
    return;
  }

  const maxVal = Math.max(...visibleValues);
  const padding = maxVal === 0 ? 0.1 : Math.abs(maxVal) * 0.1;

  yScale.options.suggestedMax = maxVal + padding;
  chart.update("none");
}

function updateOhlcChart(ohlcData, startDate, endDate) {
  const hasData = Array.isArray(ohlcData) && ohlcData.length > 0;
  if (!hasData) {
    if (ohlcChart) {
      ohlcChart.destroy();
      ohlcChart = null;
    }
    return;
  }

  const dataset = ohlcData.map((item) => ({
    x: new Date(item.date),
    o: Number(item.open),
    h: Number(item.high),
    l: Number(item.low),
    c: Number(item.close),
    y: Number(item.close),
  }));

  const prices = dataset.map((entry) => entry.h);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRange = maxPrice - minPrice;
  const padding = priceRange <= 0 ? maxPrice * 0.02 : priceRange * 0.05;
  const yMin = Math.max(minPrice - padding, 0);
  const yMax = maxPrice + padding;

  if (ohlcChart) {
    ohlcChart.destroy();
  }

  const getController = Chart.registry && Chart.registry.getController;
  const candlestickAvailable = typeof getController === "function" && !!getController.call(Chart.registry, "candlestick");
  const defaultMin = dataset[0]?.x?.getTime();
  const defaultMax = dataset[dataset.length - 1]?.x?.getTime();
  const parseBound = (value, fallback) => {
    if (!value) return fallback;
    const ts = Date.parse(value);
    return Number.isNaN(ts) ? fallback : ts;
  };
  const minBound = parseBound(startDate, defaultMin);
  const maxBound = parseBound(endDate, defaultMax);

  ohlcChart = new Chart(ohlcCtx, {
    type: candlestickAvailable ? "candlestick" : "line",
    data: {
      datasets: [
        {
          label: "K线",
          data: dataset,
          ...(candlestickAvailable
            ? {
                color: {
                  up: "#16a34a",
                  down: "#ef4444",
                  unchanged: "#6b7280",
                },
                parsing: {
                  xAxisKey: "x",
                  openKey: "o",
                  highKey: "h",
                  lowKey: "l",
                  closeKey: "c",
                },
              }
            : {
                borderColor: "#2563eb",
                backgroundColor: "rgba(37, 99, 235, 0.15)",
                parsing: {
                  xAxisKey: "x",
                  yAxisKey: "y",
                },
                tension: 0,
                stepped: true,
              }),
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          type: "time",
          min: minBound || undefined,
          max: maxBound || undefined,
          time: {
            tooltipFormat: "yyyy-MM-dd",
            unit: "day",
          },
          ticks: {
            maxTicksLimit: 8,
          },
        },
        y: {
          min: yMin,
          max: yMax,
          ticks: {
            callback: (value) => formatPrice(value),
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(context) {
              const { o, h, l, c } = context.raw;
              return `开:${formatPrice(o)} 高:${formatPrice(h)} 低:${formatPrice(l)} 收:${formatPrice(c)}`;
            },
          },
        },
        zoom: {
          zoom: {
            wheel: {
              enabled: isOhlcChartFocused,
              modifierKey: null,
            },
            mode: "x",
          },
          pan: {
            enabled: isOhlcChartFocused,
            mode: "x",
          },
        },
      },
    },
  });
  attachOhlcFocusHandlers();
  setOhlcChartFocus(isOhlcChartFocused);
}

function renderDailyTable() {
  tailTableBody.innerHTML = "";
  if (!dailyRecords.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = '<td colspan="7" style="text-align:center;color:var(--muted);">暂无数据</td>';
    tailTableBody.appendChild(tr);
    paginationInfo.textContent = "第 0 / 0 页";
    prevPageBtn.disabled = true;
    nextPageBtn.disabled = true;
    return;
  }

  const totalPages = Math.ceil(dailyRecords.length / PAGE_SIZE);
  currentPage = Math.min(Math.max(currentPage, 1), totalPages);
  const start = (currentPage - 1) * PAGE_SIZE;
  const pageItems = dailyRecords.slice(start, start + PAGE_SIZE);

  pageItems.forEach((row) => {
    const tr = document.createElement("tr");
    const price = formatPrice(row.close);
    const position = formatNumber(row.position, 2);
    const holdingPrice = formatPrice(row.holdingPrice);
    const totalReturn = formatPercent(row.totalReturn);
    tr.innerHTML = `
      <td>${row.date}</td>
      <td>${price}</td>
      <td>${position}</td>
      <td>${holdingPrice}</td>
      <td>${formatPercent(row.return)}</td>
      <td>${totalReturn}</td>
      <td>${row.action || "-"}</td>
    `;
    tailTableBody.appendChild(tr);
  });

  paginationInfo.textContent = `第 ${currentPage} / ${totalPages} 页（共 ${dailyRecords.length} 条）`;
  prevPageBtn.disabled = currentPage === 1;
  nextPageBtn.disabled = currentPage === totalPages;
}

async function runBacktest(event) {
  event.preventDefault();
  statusText.classList.remove("error");
  statusText.textContent = "正在回测...";
  submitBtn.disabled = true;

  const formData = new FormData(form);
  const payload = {
    ticker: formData.get("ticker").trim(),
    start: formData.get("start"),
    end: formData.get("end"),
    shortWindow: Number(formData.get("shortWindow")),
    longWindow: Number(formData.get("longWindow")),
    capital: Number(formData.get("capital")),
    costBps: Number(formData.get("costBps")),
    dataSource: formData.get("dataSource"),
    akshareAdjust: formData.get("akshareAdjust"),
    autoAdjust: formData.get("autoAdjust") !== null,
  };

  const timeoutMs = 45000;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const json = await response.json();

    if (!json.success) {
      statusText.textContent = json.error || "回测失败";
      statusText.classList.add("error");
      return;
    }

    renderMetrics(json.metrics);
    updateCharts(json.equityCurve, json.returns);
    updateOhlcChart(json.ohlc || [], json.params?.start, json.params?.end);
    dailyRecords = json.daily && Array.isArray(json.daily) && json.daily.length ? json.daily : json.tail || [];
    currentPage = 1;
    renderDailyTable();

    statusText.textContent = `完成：${json.params.ticker} ${json.params.start} → ${json.params.end}`;
  } catch (error) {
    console.error(error);
    if (error.name === "AbortError") {
      statusText.textContent = `请求超时（>${timeoutMs / 1000}s），请检查后端或调整参数重试`;
    } else {
      statusText.textContent = error.message || "回测出错";
    }
    statusText.classList.add("error");
  } finally {
    clearTimeout(timeoutId);
    submitBtn.disabled = false;
  }
}

form.addEventListener("submit", runBacktest);

function changePage(delta) {
  if (!dailyRecords.length) {
    return;
  }
  const totalPages = Math.ceil(dailyRecords.length / PAGE_SIZE);
  const nextPage = Math.min(Math.max(currentPage + delta, 1), totalPages);
  if (nextPage !== currentPage) {
    currentPage = nextPage;
    renderDailyTable();
  }
}

prevPageBtn.addEventListener("click", () => changePage(-1));
nextPageBtn.addEventListener("click", () => changePage(1));

statusText.textContent = "填写参数后点击“运行回测”。";
renderDailyTable();
