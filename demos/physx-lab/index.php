<?php
declare(strict_types=1);

const LAB_ROOT = __DIR__ . '/../..';
const LAB_OUT = __DIR__ . '/out';
const LAB_FIELDS = ['px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'];

function labScenes(): array {
    return [
        'rolling' => ['id'=>'rolling', 'label'=>'Rolling sphere', 'steps'=>240, 'bodies'=>1,
            'shape'=>'sphere', 'ground'=>'plane', 'rel'=>1e-3,
            'args'=>['--sphere', '--friction']],
        'mesh' => ['id'=>'mesh', 'label'=>'Mesh crossing', 'steps'=>30, 'bodies'=>1,
            'shape'=>'sphere', 'ground'=>'trimesh', 'rel'=>1e-3,
            'args'=>['--sphere', '--trimesh', '--frictionless']],
        'boxes' => ['id'=>'boxes', 'label'=>'Two-box stack', 'steps'=>30, 'bodies'=>2,
            'shape'=>'box', 'ground'=>'plane', 'rel'=>2e-2,
            'args'=>['--box', '--stacked', '--bodies', '2', '--frictionless']],
    ];
}

function labPaths(): array {
    $root = realpath(LAB_ROOT);
    return ['root'=>$root,
        'executable'=>"$root/build/physx-cumetal-runtime/artifacts/bin/UNKNOWN/release/SnippetHelloGRB",
        'runtime'=>"$root/build/libcumetal.dylib",
        'kernels'=>"$root/build/physx-cumetal-runtime/sdk_cumetal_gpu_source_bin/kernels"];
}

function labLoadTsv(string $path, int $steps, int $bodies): array {
    if (!is_file($path)) throw new RuntimeException('Missing native capture.');
    $stream = fopen($path, 'r');
    if (!$stream) throw new RuntimeException('Missing native capture.');
    try {
        $expected = array_merge($bodies === 1 ? ['step'] : ['step', 'body'], LAB_FIELDS);
        if (fgetcsv($stream, 0, "\t", '"', '') !== $expected) {
            throw new RuntimeException('Native capture has an unexpected schema.');
        }
        $frames = [];
        for ($step = 0; $step <= $steps; $step++) {
            $frame = [];
            for ($body = 0; $body < $bodies; $body++) {
                $row = fgetcsv($stream, 0, "\t", '"', '');
                if (!$row || count($row) !== count($expected) || $row[0] !== (string)$step
                    || ($bodies > 1 && $row[1] !== (string)$body)) {
                    throw new RuntimeException('Native capture has missing, duplicate, or reordered states.');
                }
                $state = [];
                foreach (array_slice($row, $bodies === 1 ? 1 : 2) as $value) {
                    if (!is_numeric($value) || !is_finite((float)$value)) {
                        throw new RuntimeException('Native capture contains non-finite state.');
                    }
                    $state[] = (float)$value;
                }
                $norm = array_sum(array_map(fn($v) => $v * $v, array_slice($state, 3, 4)));
                if (abs($norm - 1) > 0.01) throw new RuntimeException('Invalid orientation in native capture.');
                $frame[] = $state;
            }
            $frames[] = $frame;
        }
        if (fgetcsv($stream, 0, "\t", '"', '') !== false) {
            throw new RuntimeException('Native capture contains extra states.');
        }
        return $frames;
    } finally { fclose($stream); }
}

function labCompare(array $cpu, array $gpu, float $relative): array {
    if (!$cpu || count($cpu) !== count($gpu)) throw new RuntimeException('Capture length mismatch.');
    $position = 0.0;
    $scaled = 0.0;
    foreach ($cpu as $step => $bodies) {
        if (!$bodies || count($bodies) !== count($gpu[$step])) throw new RuntimeException('Body count mismatch.');
        foreach ($bodies as $body => $state) {
            $actual = $gpu[$step][$body];
            if (count($state) !== 13 || count($actual) !== 13) throw new RuntimeException('State width mismatch.');
            $distance = 0.0;
            foreach ($state as $field => $value) {
                if (!is_finite($value) || !is_finite($actual[$field])) throw new RuntimeException('Non-finite comparison.');
                $error = abs($actual[$field] - $value);
                $ratio = $error / (1e-5 + $relative * max(abs($value), 1.0));
                $scaled = max($scaled, $ratio);
                if ($field < 3) $distance += $error ** 2;
                if ($ratio > 1) throw new RuntimeException("CPU/GPU comparison failed at step $step, body $body, field " . LAB_FIELDS[$field] . '.');
            }
            $position = max($position, sqrt($distance));
        }
    }
    return ['maxPositionError'=>$position, 'maxScaledError'=>$scaled, 'comparisonPassed'=>true];
}

function labCheckLog(string $log, array $scene, bool $gpu): array {
    if (!str_contains($log, 'SnippetHelloGRB done.') ||
        !str_contains($log, 'CuMetal GRB mode: ' . ($gpu ? 'gpu' : 'cpu')) ||
        preg_match('/internal error|failed to create compute pipeline|GPU Hang|CuMetal GRB ERROR/i', $log)) {
        throw new RuntimeException('Native run did not complete successfully; inspect its retained log.');
    }
    $kernels = [];
    $device = null;
    $count = 0;
    foreach (explode("\n", $log) as $line) {
        if (!str_starts_with($line, 'CUMETAL_PROVENANCE ')) continue;
        preg_match_all('/([a-z_]+)=("[^"]*"|[^\s]+)/', $line, $matches, PREG_SET_ORDER);
        $fields = [];
        foreach ($matches as $match) $fields[$match[1]] = trim($match[2], '"');
        if (($fields['event'] ?? '') !== 'kernel_launch') continue;
        if (!$gpu || ($fields['launch_success'] ?? '') !== 'true' ||
            ($fields['device'] ?? '') !== 'apple_gpu' || ($fields['source'] ?? '') !== 'metallib' ||
            ($fields['semantic_quality'] ?? '') !== 'exact' ||
            !str_starts_with($fields['device_name'] ?? '', 'Apple ')) {
            throw new RuntimeException('Native trace contains missing, failed, or non-Apple-GPU work.');
        }
        if ($device !== null && $device !== $fields['device_name']) throw new RuntimeException('Device identity changed.');
        $device = $fields['device_name'];
        $kernels[$fields['kernel'] ?? ''] = true;
        $count++;
    }
    if ($gpu) {
        $required = ['integrateCoreParallelLaunch', 'solveStaticBlock'];
        if ($scene['ground'] === 'trimesh') {
            $required = array_merge($required, ['midphaseGeneratePairs', 'sphereTrimeshNarrowphase',
                'sortTriangleIndices', 'convexTrimeshPostProcess', 'convexTrimeshCorrelate', 'convexTrimeshFinishContacts']);
        } elseif ($scene['shape'] === 'box') {
            $required = array_merge($required, ['boxBoxNphase_Kernel', 'ZeroBodies', 'solveBlockPartition', 'writeBackBodies']);
        } else { $required[] = 'sphereNphase_Kernel'; }
        foreach ($required as $kernel) if (!isset($kernels[$kernel])) {
            throw new RuntimeException("Missing required GPU kernel: $kernel");
        }
        if ($count < $scene['steps']) throw new RuntimeException('Too few GPU completion records.');
    }
    return ['launches'=>$count, 'device'=>$device];
}

function labExecute(array $command, array $environment, string $log, float $timeout = 45): void {
    $process = proc_open($command, [0=>['file','/dev/null','r'], 1=>['file',$log,'w'], 2=>['redirect',1]],
        $pipes, realpath(LAB_ROOT), $environment, ['bypass_shell'=>true]);
    if (!is_resource($process)) throw new RuntimeException('Unable to launch native PhysX.');
    $started = microtime(true);
    $exit = -1;
    try {
        while (true) {
            $status = proc_get_status($process);
            if (!$status['running']) { $exit = $status['exitcode']; break; }
            clearstatcache(true, $log);
            if (microtime(true) - $started > $timeout || (is_file($log) && filesize($log) > 16 * 1024 * 1024)) {
                throw new RuntimeException('Native run exceeded its time or output limit.');
            }
            usleep(50000);
        }
    } finally {
        $status = proc_get_status($process);
        if ($status['running']) {
            proc_terminate($process);
            usleep(150000);
            if (proc_get_status($process)['running']) proc_terminate($process, 9);
        }
        $closed = proc_close($process);
        if ($exit === -1 && $closed >= 0) $exit = $closed;
    }
    clearstatcache(true, $log);
    if (is_file($log) && filesize($log) > 16 * 1024 * 1024) {
        throw new RuntimeException('Native run exceeded its time or output limit.');
    }
    if ($exit !== 0) throw new RuntimeException("Native PhysX exited $exit; inspect its retained log.");
}

function labCreateDirectory(string $path, bool $reuse = false): void {
    if ($reuse && is_dir($path)) return;
    if (!@mkdir($path, 0700, $reuse) && !($reuse && is_dir($path))) {
        throw new RuntimeException('Cannot create capture directory.');
    }
}

function labPersistCapture(string $directory, string $output, string $json): void {
    $length = strlen($json);
    if (@file_put_contents("$directory/result.json", $json) !== $length) {
        throw new RuntimeException('Cannot write completed capture; latest capture was not changed.');
    }
    $temporary = $output . '/latest-' . bin2hex(random_bytes(8)) . '.tmp';
    try {
        if (@file_put_contents($temporary, $json) !== $length) {
            throw new RuntimeException('Cannot stage latest capture; previous capture was not changed.');
        }
        if (!@rename($temporary, $output . '/latest.json')) {
            throw new RuntimeException('Cannot publish latest capture; previous capture was not changed.');
        }
    } finally {
        if (is_file($temporary)) @unlink($temporary);
    }
}

function labRun(string $selection): array {
    $scene = labScenes()[$selection] ?? throw new InvalidArgumentException('Unknown scene.');
    $paths = labPaths();
    if (!is_executable($paths['executable']) || !is_file($paths['runtime']) || !is_dir($paths['kernels'])) {
        throw new RuntimeException('Build the patched PhysX runtime first.');
    }
    labCreateDirectory(LAB_OUT, true);
    $lock = @fopen(LAB_OUT . '/run.lock', 'c');
    if (!$lock) throw new RuntimeException('Cannot open simulation lock.');
    if (!flock($lock, LOCK_EX | LOCK_NB)) {
        fclose($lock);
        throw new RuntimeException('A simulation is already running.');
    }
    try {
        $id = gmdate('Ymd-His') . '-' . bin2hex(random_bytes(4));
        $directory = LAB_OUT . '/' . $id;
        labCreateDirectory($directory);
        $identity = ['runtime'=>hash_file('sha256', $paths['runtime']), 'executable'=>hash_file('sha256', $paths['executable'])];
        $environment = getenv();
        foreach (array_keys($environment) as $key) if (str_starts_with($key, 'CUMETAL_')) unset($environment[$key]);
        $environment['DYLD_LIBRARY_PATH'] = $paths['root'] . '/build';
        $environment['CUMETAL_ENABLE_LLMC_CPU_EMULATION'] = '0';
        $environment['CUMETAL_DISABLE_LLMC_EMULATION'] = '1';
        $gpuEnvironment = array_merge($environment, [
            'CUMETAL_USE_METAL_DEVICE_ADDRESSES'=>'1', 'CUMETAL_PHYSX_KERNEL_DIR'=>$paths['kernels'],
            'CUMETAL_SYNC_EACH_LAUNCH'=>'1', 'CUMETAL_TRACE_GPU'=>'1',
            'CUMETAL_TOOL_XCRUN'=>$paths['root'] . '/local/toolchain/xcrun']);
        $started = microtime(true);
        $states = [];
        $traces = [];
        foreach ($selection === 'rolling' ? ['cpu','gpu','control'] : ['cpu','gpu'] as $mode) {
            $args = $scene['args'];
            if ($mode === 'control') $args = array_map(fn($arg) => $arg === '--friction' ? '--frictionless' : $arg, $args);
            $tsv = "$directory/$mode.tsv";
            $log = "$directory/$mode.log";
            labExecute(array_merge([$paths['executable'], $mode === 'cpu' ? '--cpu' : '--gpu'], $args,
                ['--steps', (string)$scene['steps'], '--dump', $tsv]), $mode === 'cpu' ? $environment : $gpuEnvironment, $log);
            $states[$mode] = labLoadTsv($tsv, $scene['steps'], $scene['bodies']);
            $traces[$mode] = labCheckLog(file_get_contents($log), $scene, $mode !== 'cpu');
        }
        $comparison = labCompare($states['cpu'], $states['gpu'], $scene['rel']);
        if (isset($states['control'])) {
            $final = $states['gpu'][$scene['steps']][0];
            $control = $states['control'][$scene['steps']][0];
            if (abs($control[7] - 5) > 0.02 || abs($control[12]) > 0.02 ||
                abs($final[7] + $final[12]) > 0.02 || $final[7] <= 0 ||
                $control[0] - $final[0] < 1) throw new RuntimeException('Friction/control invariants failed.');
        }
        if ($selection === 'mesh' && !($states['gpu'][0][0][0] < 0 && $states['gpu'][$scene['steps']][0][0] > 0)) {
            throw new RuntimeException('Sphere did not cross the mesh seam.');
        }
        foreach (['runtime','executable'] as $name) if (hash_file('sha256', $paths[$name]) !== $identity[$name]) {
            throw new RuntimeException('A native artifact changed during simulation.');
        }
        $frames = [];
        for ($i=0; $i <= $scene['steps']; $i++) $frames[] = [
            'gpu'=>$states['gpu'][$i], 'cpu'=>$states['cpu'][$i], 'control'=>$states['control'][$i] ?? null];
        $result = ['id'=>$id, 'scene'=>$scene, 'fps'=>60, 'frames'=>$frames,
            'metrics'=>array_merge($comparison, ['gpuLaunches'=>$traces['gpu']['launches'],
                'controlLaunches'=>$traces['control']['launches'] ?? 0, 'device'=>$traces['gpu']['device'],
                'elapsedSeconds'=>microtime(true)-$started]), 'identity'=>$identity, 'createdAt'=>gmdate(DATE_ATOM)];
        $json = json_encode($result, JSON_THROW_ON_ERROR | JSON_UNESCAPED_SLASHES);
        labPersistCapture($directory, LAB_OUT, $json);
        return $result;
    } finally { flock($lock, LOCK_UN); fclose($lock); }
}

function labJson(array $value, int $status = 200): never {
    http_response_code($status);
    header('Content-Type: application/json');
    header('Cache-Control: no-store');
    echo json_encode($value, JSON_THROW_ON_ERROR | JSON_UNESCAPED_SLASHES);
    exit;
}

if (defined('PHYSX_LAB_TEST')) return;

// The router exposes only this page, three library modules, and bounded run APIs.
$host = $_SERVER['HTTP_HOST'] ?? '';
if (!preg_match('/^(127\.0\.0\.1|localhost)(:\d{1,5})?$/D', $host)) labJson(['error'=>'Loopback host required.'], 403);
header('X-Content-Type-Options: nosniff');
header('Referrer-Policy: no-referrer');
header('Content-Security-Policy: default-src \'self\'; script-src \'self\' \'unsafe-inline\'; style-src \'self\' \'unsafe-inline\'; img-src \'self\' data:; connect-src \'self\'; frame-ancestors \'none\'; base-uri \'none\'; form-action \'none\'');
$path = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);
$assets = ['/assets/three/three.module.js'=>'three/build/three.module.js',
    '/assets/three/three.core.js'=>'three/build/three.core.js',
    '/assets/three/OrbitControls.js'=>'three/examples/jsm/controls/OrbitControls.js',
    '/assets/lucide.js'=>'lucide/dist/umd/lucide.min.js'];
if (isset($assets[$path])) {
    $asset = __DIR__ . '/node_modules/' . $assets[$path];
    if (!is_file($asset)) labJson(['error'=>'Frontend dependencies missing. Run npm ci --ignore-scripts.'], 503);
    header('Content-Type: application/javascript');
    readfile($asset);
    exit;
}
if ($path !== '/' && $path !== '/index.php') labJson(['error'=>'Not found.'], 404);
session_start(['cookie_httponly'=>true, 'cookie_samesite'=>'Strict', 'use_strict_mode'=>true]);
$_SESSION['physx_token'] ??= bin2hex(random_bytes(24));
$token = $_SESSION['physx_token'];
session_write_close();
$action = $_GET['action'] ?? null;
if ($action !== null) {
    try {
        if ($action === 'status' && $_SERVER['REQUEST_METHOD'] === 'GET') {
            $paths = labPaths();
            $ready = is_executable($paths['executable']) && is_file($paths['runtime']) && is_dir($paths['kernels']);
            labJson(['ready'=>$ready, 'device'=>'Apple Silicon', 'version'=>'PhysX 5.6.1', 'scenes'=>array_values(labScenes()),
                'error'=>$ready ? null : 'Patched PhysX build missing.']);
        }
        if ($action === 'latest' && $_SERVER['REQUEST_METHOD'] === 'GET') {
            if (!is_file(LAB_OUT . '/latest.json')) labJson(['error'=>'No captured run yet.'], 404);
            labJson(json_decode(file_get_contents(LAB_OUT . '/latest.json'), true, 512, JSON_THROW_ON_ERROR));
        }
        if ($action !== 'run') labJson(['error'=>'Unknown action.'], 404);
        if ($_SERVER['REQUEST_METHOD'] !== 'POST') labJson(['error'=>'POST required.'], 405);
        if (!hash_equals($token, $_SERVER['HTTP_X_PHYSX_TOKEN'] ?? '') ||
            (isset($_SERVER['HTTP_ORIGIN']) && $_SERVER['HTTP_ORIGIN'] !== 'http://' . $host)) {
            labJson(['error'=>'Same-origin session token required.'], 403);
        }
        $body = file_get_contents('php://input', false, null, 0, 4097);
        if (strlen($body) > 4096) labJson(['error'=>'Request too large.'], 413);
        $input = json_decode($body, true, 8, JSON_THROW_ON_ERROR);
        if (!is_array($input) || array_keys($input) !== ['scene'] || !is_string($input['scene']) || !isset(labScenes()[$input['scene']])) {
            labJson(['error'=>'Choose a supported scene.'], 400);
        }
        labJson(labRun($input['scene']));
    } catch (JsonException $e) { labJson(['error'=>'Invalid JSON.'], 400);
    } catch (Throwable $e) { labJson(['error'=>$e->getMessage()], 422); }
}
header('Content-Type: text/html; charset=utf-8');
header('Cache-Control: no-store');
?>
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<script>window.PHYSX_TOKEN = <?= json_encode($token, JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT) ?>;</script>
<title>PhysX on Metal | CUDA-M</title>
<link rel="icon" href="data:,">
<style>
:root{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#242b29;background:#edf0ee;font-size:14px;letter-spacing:0;--line:#dce2dd;--muted:#66746b;--green:#28764d;--coral:#be6455;--gold:#987721}
*{box-sizing:border-box}body{margin:0}button,select,input{font:inherit}button,select{color:inherit}button{cursor:pointer}button:disabled,select:disabled{cursor:wait;opacity:.45}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #80ab92;outline-offset:3px}button svg{width:18px;height:18px;flex:none}button{display:inline-flex;align-items:center;justify-content:center;gap:8px;border:1px solid var(--line);background:#fff;border-radius:5px;height:36px;padding:0 12px;white-space:nowrap}button:hover:not(:disabled){background:#f0f4f1}button:active:not(:disabled){transform:translateY(1px)}.icon{width:36px;padding:0;flex-shrink:0}.icon[aria-pressed=true]{color:var(--green);background:#e9f2eb;border-color:#a7c5b0}.shell{height:100vh;height:100dvh;min-height:560px;display:flex;flex-direction:column}.topbar{height:66px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:#fff;border-bottom:1px solid var(--line);gap:16px}.brand{display:flex;align-items:center;gap:12px;min-width:0}.brand-mark{height:30px;width:30px;color:var(--green);flex:none}.brand h1{font-size:18px;line-height:1.2;font-weight:650;margin:0}.brand small{display:block;color:var(--muted);font-size:10px;font-weight:650;margin-top:4px;text-transform:uppercase;letter-spacing:1.1px}.backend{display:flex;align-items:center;gap:8px;font-size:12px;white-space:nowrap;color:var(--muted)}.dot{width:6px;height:6px;display:inline-block;background:#8da096;border-radius:50%;flex:none}.dot.ready{background:#388655}.controls{padding:12px 24px;display:flex;align-items:center;gap:10px;background:#fff;border-bottom:1px solid var(--line);flex:none}.scene-field{display:flex;align-items:center;gap:10px}.field-label{font-size:11px;font-weight:650;text-transform:uppercase;color:var(--muted);letter-spacing:.8px}select{height:36px;border:1px solid var(--line);background:#fff;border-radius:5px;padding:0 28px 0 10px;max-width:100%}#scene-select{width:178px}.run{background:#28764d;border-color:#28764d;color:#fff;font-weight:600}.run:hover:not(:disabled){background:#205e3e}.capture-state{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12px;margin-left:auto;white-space:nowrap}.capture-state.busy .dot{background:#bc9134;animation:pulse 1.2s infinite}.capture-state.error{color:#a4473d}.capture-state.error .dot{background:#a4473d}@keyframes pulse{50%{opacity:.25}}.stage{flex:1;min-height:200px;position:relative;overflow:hidden;background:#edf0ee;isolation:isolate}.stage canvas{width:100%;height:100%;display:block;touch-action:none}.legend{position:absolute;left:24px;top:20px;display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:#4d5952;pointer-events:none;z-index:2}.legend span{display:flex;align-items:center;gap:6px}.swatch{width:9px;height:9px;border-radius:2px;background:var(--green);display:inline-block}.swatch.control{background:var(--gold)}.swatch.cpu{background:transparent;border:1.5px solid var(--coral)}.view-tools{display:flex;gap:6px;position:absolute;right:24px;top:14px;z-index:3}.view-tools button{background:#ffffffde}.run-caption{position:absolute;left:24px;bottom:18px;pointer-events:none;display:flex;flex-direction:column;gap:4px;color:#6b786f;font-size:11px;line-height:1.4}.run-caption strong{font-weight:550;color:#38483f;font-size:12px}.lane-note{position:absolute;right:24px;bottom:18px;font-size:11px;color:#6b786f;pointer-events:none}.empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:12px;pointer-events:none;padding:30px;text-align:center}.empty svg{width:28px;height:28px;color:#81918a}.empty p{margin:0;color:#5d6a62;font-size:13px;max-width:400px;line-height:1.6}.hidden{display:none!important}.error-banner{background:#fff0ec;border-bottom:1px solid #edc6bc;color:#984539;padding:10px 24px;font-size:12px;line-height:1.6;flex:none;overflow-wrap:anywhere;max-height:116px;overflow:auto}.transport{display:flex;align-items:center;gap:10px;padding:12px 24px;background:#fff;border-top:1px solid var(--line);flex:none}.play{background:#edf4ef;border-color:#c4d7ca;color:var(--green)}.time{font-variant-numeric:tabular-nums;font-size:12px;color:#617168;white-space:nowrap;min-width:90px;text-align:center}#timeline{flex:1;min-width:40px;accent-color:var(--green);height:24px;cursor:pointer}#speed{width:74px;flex:none}.reference{display:flex;align-items:center;gap:6px;white-space:nowrap;font-size:12px;color:#526056;padding-left:10px}.reference input{accent-color:var(--coral);width:15px;height:15px;cursor:pointer}.metrics{display:grid;grid-template-columns:1.2fr 1fr 1fr 1fr;border-top:1px solid var(--line);background:#fff;padding:14px 24px 17px;flex:none;gap:24px}.metric{min-width:0}.metric label{display:block;color:#718076;font-size:10px;text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px}.metric-value{display:flex;align-items:baseline;gap:6px;min-height:18px;white-space:nowrap;font-size:14px;font-weight:550;font-variant-numeric:tabular-nums}.metric-value small{font-size:11px;font-weight:400;color:#718076}.pass{color:var(--green)}.fail{color:#a4473d}.footer-note{padding:7px 24px;background:#fafbf9;border-top:1px solid #e6eae6;display:flex;justify-content:space-between;gap:12px;flex:none;font-size:10px;color:#7c877e}.footer-note a{color:inherit;text-decoration:none}.footer-note a:hover{text-decoration:underline}
.brand small,.field-label,.metric label{letter-spacing:0}
@media(max-width:760px){.topbar{padding:0 16px;height:62px}.brand h1{font-size:16px}.brand small{font-size:9px}.backend{font-size:11px}.controls{padding:10px 16px;flex-wrap:wrap;gap:8px}.field-label{display:none}#scene-select{width:160px}.capture-state{font-size:11px}.legend{left:16px;top:17px;gap:10px;font-size:11px;max-width:calc(100% - 110px)}.view-tools{right:16px;top:10px}.run-caption{left:16px;bottom:13px}.lane-note{right:16px;bottom:13px;max-width:120px;text-align:right;font-size:10px}.transport{padding:10px 16px;gap:7px;flex-wrap:wrap}#timeline{order:5;flex-basis:calc(100% - 104px)}.time{order:6;min-width:94px}.reference{margin-left:auto;padding:0}.metrics{padding:12px 16px;gap:14px;grid-template-columns:1fr 1fr}.metric label{font-size:9px;margin-bottom:4px}.metric-value{font-size:13px}.footer-note{padding:7px 16px;font-size:9px}.footer-note span:last-child{display:none}}
@media(max-width:410px){.backend #device-name{max-width:75px;white-space:normal;text-align:right}.brand{gap:8px}.brand-mark{width:25px;height:25px}.controls .capture-state{width:100%;margin:0;padding-top:2px}#scene-select{width:174px}.run{margin-left:auto}.legend{max-width:calc(100% - 100px);gap:6px 10px}.legend span{font-size:10px}.view-tools{gap:4px}.view-tools .icon{height:32px;width:32px}.run-caption{max-width:60%}.reference{font-size:11px}.stage{min-height:220px}}
@media(max-width:760px){#timeline{flex:1 1 calc(100% - 108px);margin:0}.time{flex:0 0 94px;min-width:94px}}
</style>
<script type="importmap">{"imports":{"three":"/assets/three/three.module.js"}}</script>
<script src="/assets/lucide.js"></script>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand"><i data-lucide="box" class="brand-mark"></i><div><h1>PhysX on Metal</h1><small>CUDA-M / Native GPU Lab</small></div></div>
    <div class="backend"><span class="dot" id="backend-dot"></span><span id="device-name">Checking backend</span></div>
  </header>
  <section class="controls" aria-label="Simulation controls">
    <div class="scene-field"><label class="field-label" for="scene-select">Scene</label><select id="scene-select" aria-label="Scene"><option value="rolling">Rolling sphere</option><option value="mesh">Mesh crossing</option><option value="boxes">Two-box stack</option></select></div>
    <button class="run" id="run" disabled><i data-lucide="cpu"></i>Run on GPU</button>
    <div class="capture-state" id="capture-state" role="status"><span class="dot"></span><span id="status-text">Connecting</span></div>
  </section>
  <div class="error-banner hidden" id="error" role="alert"></div>
  <main class="stage" id="stage" aria-label="Recorded PhysX simulation viewport">
    <div class="legend"><span><i class="swatch"></i>CUDA-M GPU</span><span id="control-legend" class="hidden"><i class="swatch control"></i>Frictionless GPU</span><span id="cpu-legend" class="hidden"><i class="swatch cpu"></i>CPU reference</span></div>
    <div class="view-tools"><button class="icon" id="grid" title="Toggle ground grid" aria-label="Toggle ground grid" aria-pressed="true"><i data-lucide="grid-3x3"></i></button><button class="icon" id="camera" title="Reset camera" aria-label="Reset camera"><i data-lucide="scan"></i></button></div>
    <div class="empty" id="empty"><i data-lucide="box"></i><p id="empty-text">Preparing Apple GPU</p></div>
    <div class="run-caption"><strong id="scene-name">PhysX 5.6.1</strong><span id="capture-detail">Source-recompiled CUDA kernels</span></div>
    <div class="lane-note hidden" id="lane-note">Lanes offset for comparison</div>
  </main>
  <section class="transport" aria-label="Capture playback">
    <button class="icon play" id="play" title="Play capture" aria-label="Play capture" disabled><i data-lucide="play"></i></button>
    <button class="icon" id="replay" title="Replay capture" aria-label="Replay capture" disabled><i data-lucide="rotate-ccw"></i></button>
    <span class="time" id="time">0.00 / 0.00 s</span>
    <input id="timeline" type="range" min="0" max="1" value="0" step="1" aria-label="Playback frame" disabled>
    <select id="speed" aria-label="Playback speed"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select>
    <label class="reference"><input type="checkbox" id="cpu">CPU reference</label>
  </section>
  <section class="metrics" aria-label="Measured simulation results">
    <div class="metric"><label>Apple GPU dispatches</label><div class="metric-value"><span id="launches">-</span><small id="control-launches"></small></div></div>
    <div class="metric"><label>CPU comparison</label><div class="metric-value" id="comparison">-</div></div>
    <div class="metric"><label>Max position error</label><div class="metric-value"><span id="position-error">-</span><small>m</small></div></div>
    <div class="metric"><label>Native run time</label><div class="metric-value"><span id="elapsed">-</span><small>s</small></div></div>
  </section>
  <footer class="footer-note"><span>Bounded PhysX source port &middot; Recorded native results</span><span>CUDA &rarr; CuMetal &rarr; Apple GPU</span></footer>
</div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from '/assets/three/OrbitControls.js';

const $ = id => document.getElementById(id);
const icons = () => window.lucide?.createIcons();
icons();
const state = {data:null,frame:0,playing:false,busy:false,ready:false,renderCount:0,errors:[],status:null};
const stage = $('stage');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xedf0ee);
const camera = new THREE.PerspectiveCamera(37,1,.1,250);
const renderer = new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.domElement.setAttribute('aria-label','Three-dimensional recorded PhysX body motion');
stage.prepend(renderer.domElement);
const orbit = new OrbitControls(camera,renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = .08;
orbit.maxPolarAngle = Math.PI*.49;
orbit.minDistance = 5;
orbit.maxDistance = 90;
scene.add(new THREE.HemisphereLight(0xffffff,0x91a297,2.4));
const sun = new THREE.DirectionalLight(0xffffff,3.1);
sun.position.set(-4,12,7);
sun.castShadow = true;
sun.shadow.mapSize.set(2048,2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -22;
sun.shadow.camera.right = sun.shadow.camera.top = 22;
sun.shadow.normalBias = .025;
scene.add(sun);
scene.add(sun.target);
const world = new THREE.Group();
scene.add(world);
const bodyGroups = {gpu:[],cpu:[],control:[]};
const trails = [];
let grid = null;
let target = new THREE.Vector3(3,1,0);
let extent = 13;
let lastTime = performance.now();
let lastDrawnFrame = -1;
let narrowViewport = stage.clientWidth < 600;

function ballTexture(color){
  const canvas = document.createElement('canvas');canvas.width=512;canvas.height=256;
  const ctx=canvas.getContext('2d');ctx.fillStyle=color;ctx.fillRect(0,0,512,256);
  ctx.fillStyle='#e7eee8';ctx.fillRect(0,113,512,30);
  for(let x=0;x<512;x+=128){ctx.fillRect(x,0,10,256);ctx.beginPath();ctx.arc(x+64,66,14,0,Math.PI*2);ctx.fill();}
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  texture.anisotropy=renderer.capabilities.getMaxAnisotropy();return texture;
}
const maps={gpu:ballTexture('#3b8960'),control:ballTexture('#b6963c')};
const materials={gpu:new THREE.MeshStandardMaterial({map:maps.gpu,roughness:.53,metalness:.08}),control:new THREE.MeshStandardMaterial({map:maps.control,roughness:.53,metalness:.08}),cpu:new THREE.MeshBasicMaterial({color:0xbe6455,wireframe:true,transparent:true,opacity:.36,depthWrite:false})};
const boxMaterials={gpu:new THREE.MeshStandardMaterial({color:0x3b8960,roughness:.52,metalness:.05}),control:new THREE.MeshStandardMaterial({color:0xb6963c,roughness:.52,metalness:.05})};
const sphereGeometry=new THREE.SphereGeometry(1,40,24);
const ghostSphereGeometry=new THREE.SphereGeometry(1.008,20,12);
const boxGeometry=new THREE.BoxGeometry(2,2,2);
const edgeMaterial=new THREE.LineBasicMaterial({color:0xe6eee7,transparent:true,opacity:.7});

function disposeWorld(){
  world.traverse(object=>{if(object.userData.disposableGeometry)object.geometry?.dispose();if(object.userData.disposableMaterial)object.material?.dispose();});
  world.clear();Object.values(bodyGroups).forEach(list=>list.length=0);trails.length=0;
}
function laneOffset(key){return state.data?.frames[0].control ? (key==='control' ? 2.25 : -2.25) : 0;}
function setupWorld(){
  disposeWorld();
  const data=state.data;
  const bounds=new THREE.Box3();
  for(const frame of data.frames)for(const key of ['gpu','control'])for(const pose of frame[key]||[])bounds.expandByPoint(new THREE.Vector3(pose[0],pose[1],pose[2]+laneOffset(key)));
  const size=bounds.getSize(new THREE.Vector3());
  target.copy(bounds.getCenter(new THREE.Vector3()));target.y=Math.max(1,target.y*.65);
  extent=Math.max(11,size.x+5,size.y+5,size.z+5);
  const floorSize=Math.max(40,extent*2.4);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(floorSize,floorSize),new THREE.MeshStandardMaterial({color:0xe1e7e1,roughness:.98}));
  floor.rotation.x=-Math.PI/2;floor.position.set(target.x,-.015,0);floor.receiveShadow=true;
  floor.userData={disposableGeometry:true,disposableMaterial:true};world.add(floor);
  grid=new THREE.GridHelper(floorSize,Math.ceil(floorSize),0xbac7bd,0xd0dbd1);grid.position.set(Math.round(target.x),.003,0);grid.material.transparent=true;grid.material.opacity=.64;
  grid.userData={disposableGeometry:true,disposableMaterial:true};grid.visible=$('grid').getAttribute('aria-pressed')==='true';world.add(grid);
  if(data.scene.id==='mesh'){
    const diagonal=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-20,.011,20),new THREE.Vector3(20,.011,-20)]);
    const seam=new THREE.Line(diagonal,new THREE.LineDashedMaterial({color:0x9aaea0,dashSize:.25,gapSize:.13}));
    seam.computeLineDistances();seam.userData={disposableGeometry:true,disposableMaterial:true};world.add(seam);
  }
  for(const key of ['gpu','cpu','control']){
    if(!data.frames[0][key])continue;
    for(let i=0;i<data.scene.bodies;i++){
      const isBox=data.scene.shape==='box'||data.scene.id==='boxes';
      const body=new THREE.Mesh(isBox?boxGeometry:key==='cpu'?ghostSphereGeometry:sphereGeometry,key==='cpu'?materials.cpu:isBox?boxMaterials[key]:materials[key]);
      body.castShadow=key!=='cpu';body.receiveShadow=key!=='cpu';
      if(isBox&&key!=='cpu'){
        const edges=new THREE.LineSegments(new THREE.EdgesGeometry(boxGeometry),edgeMaterial);edges.userData.disposableGeometry=true;body.add(edges);
      }
      if(key==='cpu'){body.scale.setScalar(isBox?1.004:1);body.visible=$('cpu').checked;}
      bodyGroups[key].push(body);world.add(body);
      if(key!=='cpu'){
        const vertices=data.frames.map(frame=>new THREE.Vector3(frame[key][i][0],.022,frame[key][i][2]+laneOffset(key)));
        const geometry=new THREE.BufferGeometry().setFromPoints(vertices);
        const trail=new THREE.Line(geometry,new THREE.LineBasicMaterial({color:key==='gpu'?0x488663:0xb69b50,transparent:true,opacity:.62}));
        trail.userData={disposableGeometry:true,disposableMaterial:true};trails.push(trail);world.add(trail);
      }
    }
  }
  sun.position.set(target.x-4,12,7);sun.target.position.set(target.x,0,0);
  $('control-legend').classList.toggle('hidden',!data.frames[0].control);
  $('lane-note').classList.toggle('hidden',!data.frames[0].control);
  lastDrawnFrame=-1;resetCamera();drawFrame();
}
function resetCamera(){
  const narrow=stage.clientWidth<600;
  const distance=extent*(narrow?1.4:.95);
  camera.position.copy(target).add(new THREE.Vector3(distance*.66,distance*.68,distance*.95));
  orbit.target.copy(target);orbit.update();
}
function resize(){
  const width=stage.clientWidth,height=stage.clientHeight;
  renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();
  if(narrowViewport!==(width<600)){narrowViewport=width<600;resetCamera();}
}
new ResizeObserver(resize).observe(stage);
function drawFrame(){
  if(!state.data)return;
  const data=state.data,index=Math.min(data.frames.length-1,Math.floor(state.frame)),next=Math.min(index+1,data.frames.length-1),fraction=state.frame-index;
  const first=data.frames[index],second=data.frames[next];
  for(const key of ['gpu','cpu','control'])bodyGroups[key].forEach((body,i)=>{
    const a=first[key][i],b=second[key][i];
    body.position.set(THREE.MathUtils.lerp(a[0],b[0],fraction),THREE.MathUtils.lerp(a[1],b[1],fraction),THREE.MathUtils.lerp(a[2],b[2],fraction)+laneOffset(key));
    body.quaternion.set(a[3],a[4],a[5],a[6]).slerp(new THREE.Quaternion(b[3],b[4],b[5],b[6]),fraction);
  });
  trails.forEach(trail=>trail.geometry.setDrawRange(0,index+1));
  if(index!==lastDrawnFrame){
    $('timeline').value=String(index);
    $('time').textContent=`${(index/data.fps).toFixed(2)} / ${((data.frames.length-1)/data.fps).toFixed(2)} s`;
    $('capture-detail').textContent=`Frame ${index} / ${data.frames.length-1} · ${data.scene.bodies} ${data.scene.bodies===1?'body':'bodies'} · ${data.fps} Hz`;
    lastDrawnFrame=index;
  }
}
function setPlaying(value){
  state.playing=value;
  const label=value?'Pause capture':'Play capture';
  $('play').setAttribute('aria-label',label);$('play').title=label;
  $('play').innerHTML=`<i data-lucide="${value?'pause':'play'}"></i>`;icons();
}
function animate(now){
  requestAnimationFrame(animate);
  const delta=Math.min((now-lastTime)/1000,.1);lastTime=now;
  if(state.playing&&state.data){
    state.frame+=delta*state.data.fps*Number($('speed').value);
    if(state.frame>=state.data.frames.length-1){state.frame=state.data.frames.length-1;setPlaying(false);}
    drawFrame();
  }
  orbit.update();renderer.render(scene,camera);state.renderCount++;
}
resize();resetCamera();requestAnimationFrame(animate);
function status(text,kind=''){$('status-text').textContent=text;$('capture-state').className=`capture-state ${kind}`;}
function showError(message){state.errors.push(String(message));$('error').textContent=String(message);$('error').classList.remove('hidden');status('Run unavailable','error');if(!state.data)$('empty-text').textContent='No verified capture available';}
function validate(data){
  if(!data||!Array.isArray(data.frames)||data.frames.length<2||data.frames.length>2001||!Number.isInteger(data.scene?.bodies)||data.scene.bodies<1||data.scene.bodies>16||!Number.isFinite(data.fps)||data.fps<=0)throw new Error('Invalid native capture metadata.');
  const hasControl=Array.isArray(data.frames[0].control);
  for(const frame of data.frames)for(const key of ['gpu','cpu',...(hasControl?['control']:[])]){
    if(!Array.isArray(frame[key])||frame[key].length!==data.scene.bodies||frame[key].some(p=>!Array.isArray(p)||p.length<7||p.some(n=>!Number.isFinite(n))))throw new Error('Native capture contains an invalid body transform.');
  }
  if(!data.metrics||!Number.isFinite(data.metrics.gpuLaunches)||data.metrics.gpuLaunches<=0||!Number.isFinite(data.metrics.maxPositionError)||!Number.isFinite(data.metrics.elapsedSeconds))throw new Error('Native capture is missing measured GPU evidence.');
}
function loadCapture(data){
  validate(data);state.data=data;state.frame=0;
  $('scene-select').value=data.scene.id;$('scene-name').textContent=data.scene.label;
  $('empty').classList.add('hidden');$('error').classList.add('hidden');
  $('timeline').max=String(data.frames.length-1);$('timeline').disabled=false;
  $('play').disabled=false;$('replay').disabled=false;
  $('launches').textContent=data.metrics.gpuLaunches.toLocaleString();
  $('control-launches').textContent=data.metrics.controlLaunches?`+ ${data.metrics.controlLaunches.toLocaleString()} control`:'';
  $('comparison').textContent=data.metrics.comparisonPassed?'Passed':'Outside tolerance';
  $('comparison').className=`metric-value ${data.metrics.comparisonPassed?'pass':'fail'}`;
  $('position-error').textContent=data.metrics.maxPositionError===0?'0':data.metrics.maxPositionError.toExponential(2);
  $('elapsed').textContent=data.metrics.elapsedSeconds.toFixed(2);
  $('device-name').textContent=data.metrics.device||state.status?.device||'Apple GPU';
  status('Captured run');setupWorld();setPlaying(true);
}
async function request(action,options={}){
  const response=await fetch(`?action=${action}`,{cache:'no-store',credentials:'same-origin',...options});
  if(action==='latest'&&response.status===404)return null;
  const data=await response.json();
  if(!response.ok)throw new Error(typeof data.error==='string'?data.error:data.error?.message||`Request failed (${response.status}).`);
  return data;
}
async function run(){
  if(state.busy||!state.ready)return;
  state.busy=true;setPlaying(false);$('run').disabled=true;$('scene-select').disabled=true;$('error').classList.add('hidden');
  status('Simulating on GPU','busy');if(!state.data)$('empty-text').textContent='Running native PhysX';
  try{loadCapture(await request('run',{method:'POST',headers:{'Content-Type':'application/json','X-PhysX-Token':window.PHYSX_TOKEN||''},body:JSON.stringify({scene:$('scene-select').value})}));}
  catch(error){showError(error.message);}
  finally{state.busy=false;$('run').disabled=!state.ready;$('scene-select').disabled=false;}
}
$('run').addEventListener('click',run);
$('play').addEventListener('click',()=>{if(!state.data)return;if(state.frame>=state.data.frames.length-1)state.frame=0;setPlaying(!state.playing);drawFrame();});
$('replay').addEventListener('click',()=>{state.frame=0;drawFrame();setPlaying(true);});
$('timeline').addEventListener('input',()=>{setPlaying(false);state.frame=Number($('timeline').value);drawFrame();});
$('camera').addEventListener('click',resetCamera);
$('grid').addEventListener('click',()=>{const visible=$('grid').getAttribute('aria-pressed')!=='true';$('grid').setAttribute('aria-pressed',String(visible));if(grid)grid.visible=visible;});
$('cpu').addEventListener('change',()=>{bodyGroups.cpu.forEach(body=>body.visible=$('cpu').checked);$('cpu-legend').classList.toggle('hidden',!$('cpu').checked);});
renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();setPlaying(false);showError('The graphics context was lost. Reload this page to restore the viewport.');});
window.physxLab=Object.freeze({getState:()=>({ready:state.ready,busy:state.busy,frame:state.frame,frameCount:state.data?.frames.length||0,scene:state.data?.scene.id||null,captureId:state.data?.id||null,playing:state.playing,renderCount:state.renderCount,canvas:{width:renderer.domElement.width,height:renderer.domElement.height},gpuPositions:bodyGroups.gpu.map(body=>body.position.toArray()),cameraPosition:camera.position.toArray(),gridVisible:grid?.visible||false,cpuVisible:$('cpu').checked,comparisonPassed:state.data?.metrics.comparisonPassed??null,errors:[...state.errors]})});
async function init(){
  try{
    state.status=await request('status');state.ready=state.status.ready===true;
    $('device-name').textContent=state.status.device||'Apple GPU';$('backend-dot').classList.toggle('ready',state.ready);
    $('run').disabled=!state.ready;
    if(!state.ready){showError(state.status.error||'Native PhysX backend is not ready.');return;}
    const latest=await request('latest');if(latest)loadCapture(latest);else await run();
  }catch(error){showError(error.message);}
}
init();
</script>
</body>
</html>
