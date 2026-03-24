// ======================== THEME TOGGLE ========================
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    
    // Toast notification for theme change
    showToast(
        next === 'dark' ? 'success' : 'info',
        'Theme Updated',
        `Switched to ${next} mode interface.`
    );
}

// Init theme
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);

// ======================== TOAST NOTIFICATIONS ========================
function showToast(type, title, message) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = 'toast';
    
    let iconClass = 'fa-info-circle';
    if(type === 'success') iconClass = 'fa-check-circle';
    if(type === 'error') iconClass = 'fa-exclamation-circle';
    if(type === 'warn') iconClass = 'fa-exclamation-triangle';
    
    toast.innerHTML = `
        <div class="toast-icon ${type}"><i class="fas ${iconClass}"></i></div>
        <div class="toast-body">
            <div class="toast-title">${title}</div>
            <div class="toast-msg">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()"><i class="fas fa-times"></i></button>
    `;
    
    container.appendChild(toast);
    
    // Auto remove
    setTimeout(() => {
        toast.classList.add('exit');
        setTimeout(() => toast.remove(), 400);
    }, 4000);
}

// ======================== CURSOR GLOW ========================
const cursorGlow = document.getElementById('cursorGlow');
let mouseX = -300, mouseY = -300;

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
});

function animateCursor() {
    cursorGlow.style.left = mouseX + 'px';
    cursorGlow.style.top = mouseY + 'px';
    requestAnimationFrame(animateCursor);
}
animateCursor();

// ======================== NAVBAR SCROLL ========================
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 40);
});

// ======================== SCROLL REVEAL ========================
const revealElements = document.querySelectorAll('.reveal');
const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

revealElements.forEach(el => revealObserver.observe(el));

// ======================== TILT EFFECT ========================
document.querySelectorAll('.liquid-glass').forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5;
        const y = (e.clientY - rect.top) / rect.height - 0.5;
        card.style.transform = `perspective(1000px) rotateY(${x * 3}deg) rotateX(${-y * 3}deg) translateY(-2px)`;
    });

    card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(1000px) rotateY(0deg) rotateX(0deg) translateY(0)';
    });
});

// ======================== DRAG & DROP ========================
const dropZone = document.getElementById('dropZone');

['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
});

['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('fileInput').files = files;
        fileSelected();
    }
});

function fileSelected() {
    const file = document.getElementById('fileInput').files[0];
    if (file) {
        const display = document.getElementById('fileDisplay');
        document.getElementById('fileName').innerText = file.name;
        display.classList.add('show');
        showToast('info', 'File Loaded', `${file.name} ready for analysis.`);
    }
}

// ======================== BUTTON RIPPLE ========================
document.querySelectorAll('.btn-primary').forEach(btn => {
    btn.addEventListener('click', function (e) {
        const rect = this.getBoundingClientRect();
        const ripple = document.createElement('span');
        ripple.className = 'ripple';
        const size = Math.max(rect.width, rect.height);
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
        ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
        this.appendChild(ripple);
        setTimeout(() => ripple.remove(), 600);
    });
});

// ======================== UPLOAD FORM ========================
document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        showToast('error', 'Missing File', 'Please select an ECG file to analyze.');
        return;
    }

    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const btnIcon = document.getElementById('btnIcon');
    const resultArea = document.getElementById('resultArea');

    analyzeBtn.disabled = true;
    btnText.innerText = "Processing Sequence...";
    btnIcon.className = "fas fa-spinner fa-spin";
    resultArea.classList.remove('show');
    
    // Reset visual elements
    document.getElementById('probNormal').style.width = '0%';
    document.getElementById('probApnea').style.width = '0%';
    document.getElementById('gaugeArc').style.strokeDashoffset = 100.5;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const res = await fetch('/upload_ecg', { method: 'POST', body: formData });
        const data = await res.json();
        
        if(data.error) throw new Error(data.error);

        // Populate elements
        const diagnosis = document.getElementById('resDiagnosis');
        const severity = document.getElementById('resSeverity');
        const confidence = document.getElementById('resConfidence');
        const accent = document.getElementById('resultAccent');
        const recText = document.getElementById('recText');

        diagnosis.innerText = data.result;
        severity.innerText = data.severity;
        Object.values(severity.classList).forEach(c => {
            if(c.startsWith('sev-')) severity.classList.remove(c);
        });
        severity.classList.add(`sev-${data.severity.toLowerCase()}`);
        
        confidence.innerText = data.confidence;
        recText.innerText = data.recommendation;

        // Colors
        if (data.result.includes("Apnea")) {
            diagnosis.className = "result-diagnosis apnea";
            accent.style.background = "var(--color-danger)";
        } else {
            diagnosis.className = "result-diagnosis normal";
            accent.style.background = "var(--color-success)";
        }

        resultArea.classList.add('show');
        
        // Detailed animation of probabilities
        setTimeout(() => {
            document.getElementById('probNormal').style.width = data.normal_probability;
            document.getElementById('probNormalVal').innerText = data.normal_probability;
            
            document.getElementById('probApnea').style.width = data.apnea_probability;
            document.getElementById('probApneaVal').innerText = data.apnea_probability;
            
            // Animate Risk Gauge
            animateGauge(data.risk_score);
        }, 300);

        showToast('success', 'Analysis Complete', 'Multi-scale ECG features extracted and classified.');
        
    } catch (err) {
        showToast('error', 'Analysis Failed', err.message);
    } finally {
        analyzeBtn.disabled = false;
        btnText.innerText = "Analyze Segment";
        btnIcon.className = "fas fa-microscope";
    }
});

// ======================== GAUGE ANIMATOR ========================
function animateGauge(targetScore) {
    // Score is 0-10. SVG dash range is 100.5 (empty) to 0 (full)
    const el = document.getElementById('gaugeValue');
    const arc = document.getElementById('gaugeArc');
    
    let start = 0;
    const duration = 1200;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (targetScore - start) * eased;

        el.innerText = current.toFixed(1);
        
        // Update SVG arc (100.5 is circumference)
        const offset = 100.5 - (current / 10) * 100.5;
        arc.style.strokeDashoffset = offset;

        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ======================== BENCHMARK ========================
async function runBenchmark() {
    const btn = document.getElementById('benchmarkBtn');
    const benchBtnText = document.getElementById('benchBtnText');
    const benchIcon = document.getElementById('benchIcon');
    const empty = document.getElementById('benchEmpty');
    const loading = document.getElementById('benchLoading');
    const results = document.getElementById('benchResults');
    const img = document.getElementById('benchImage');

    btn.disabled = true;
    benchBtnText.innerText = "Running...";
    benchIcon.className = "fas fa-spinner fa-spin";
    benchIcon.style.fontSize = "11px";

    empty.classList.add('hidden');
    results.classList.add('hidden');
    loading.classList.add('show');
    img.style.opacity = '0';
    
    showToast('info', 'Benchmark Started', 'Loading 17,075 segments from cache...');

    try {
        const res = await fetch('/run_benchmark', { method: 'POST' });
        if (!res.ok) throw new Error("Server error " + res.status);
        const data = await res.json();

        // Animate counter for metrics
        animateValue('metAcc', data.accuracy);
        animateValue('metAuc', data.auc);
        animateValue('metSens', data.sensitivity);
        animateValue('metSpec', data.specificity);

        img.src = data.image_url + "?t=" + Date.now();

        img.onload = function () {
            loading.classList.remove('show');
            results.classList.remove('hidden');
            setTimeout(() => { img.style.opacity = '1'; }, 100);
            showToast('success', 'Benchmark Evaluated', 'Test set evaluation completed successfully.');
        };

        img.onerror = function () {
            loading.classList.remove('show');
            results.classList.remove('hidden');
            showToast('warn', 'Plot Error', 'Could not load confusion matrix plot.');
        };

        setTimeout(function () {
            if (!loading.classList.contains('show')) return;
            loading.classList.remove('show');
            results.classList.remove('hidden');
            img.style.opacity = '1';
        }, 5000);

    } catch (err) {
        showToast('error', 'Benchmark Failed', err.message);
        loading.classList.remove('show');
        empty.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        benchBtnText.innerText = "Completed";
        benchIcon.className = "fas fa-check-circle";
        benchIcon.style.fontSize = "11px";
        benchIcon.style.color = "var(--color-success)";
    }
}

// ======================== ANIMATED COUNTER ========================
function animateValue(elementId, targetText) {
    const el = document.getElementById(elementId);
    const isPercent = targetText.includes('%');
    const numericValue = parseFloat(targetText);

    if (isNaN(numericValue)) {
        el.innerText = targetText;
        return;
    }

    let start = 0;
    const duration = 1200;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (numericValue - start) * eased;

        if (isPercent) {
            el.innerText = current.toFixed(2) + '%';
        } else {
            el.innerText = current.toFixed(4);
        }

        if (progress < 1) requestAnimationFrame(update);
        else el.innerText = targetText;
    }

    requestAnimationFrame(update);
}
