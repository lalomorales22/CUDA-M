<?php
declare(strict_types=1);

define('PHYSX_LAB_TEST', true);
require __DIR__ . '/index.php';

$directory = sys_get_temp_dir() . '/physx-lab-tests-' . bin2hex(random_bytes(8));
if (!mkdir($directory, 0700)) throw new RuntimeException('Cannot create test directory.');
register_shutdown_function(function () use ($directory): void {
    $files = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($directory,
        FilesystemIterator::SKIP_DOTS), RecursiveIteratorIterator::CHILD_FIRST);
    foreach ($files as $file) {
        if ($file->isDir() && !$file->isLink()) rmdir($file->getPathname());
        else unlink($file->getPathname());
    }
    rmdir($directory);
});

$passed = 0;
$failed = 0;
function check(bool $condition, string $message): void {
    if (!$condition) throw new RuntimeException($message);
}
function near(float $actual, float $expected, float $tolerance = 1e-10): void {
    check(abs($actual - $expected) <= $tolerance, "Expected $expected, received $actual.");
}
function rejects(callable $operation, string $message): void {
    try { $operation(); }
    catch (RuntimeException $error) {
        check(str_contains($error->getMessage(), $message), 'Unexpected error: ' . $error->getMessage());
        return;
    }
    throw new RuntimeException('Expected rejection containing: ' . $message);
}
function test(string $name, callable $operation): void {
    global $passed, $failed;
    try {
        $operation();
        $passed++;
        echo "PASS: $name\n";
    } catch (Throwable $error) {
        $failed++;
        fwrite(STDERR, "FAIL: $name: {$error->getMessage()}\n");
    }
}
function state(): array {
    return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0];
}
function captureRows(int $steps = 1, int $bodies = 1): array {
    $rows = [array_merge($bodies === 1 ? ['step'] : ['step', 'body'], LAB_FIELDS)];
    for ($step = 0; $step <= $steps; $step++) {
        for ($body = 0; $body < $bodies; $body++) {
            $value = state();
            $value[0] = $step + $body;
            $rows[] = array_merge($bodies === 1 ? [$step] : [$step, $body], $value);
        }
    }
    return $rows;
}
function captureFile(array $rows): string {
    global $directory;
    $path = tempnam($directory, 'capture-');
    $stream = fopen($path, 'w');
    foreach ($rows as $row) fputcsv($stream, $row, "\t", '"', '', "\n");
    fclose($stream);
    return $path;
}
function launch(string $kernel, array $overrides = []): string {
    $fields = array_merge(['event'=>'kernel_launch', 'kernel'=>$kernel, 'source'=>'metallib',
        'device'=>'apple_gpu', 'launch_success'=>'true', 'semantic_quality'=>'exact',
        'device_name'=>'Apple M4'], $overrides);
    return 'CUMETAL_PROVENANCE ' . implode(' ', array_map(
        fn($key, $value) => $key . '="' . $value . '"', array_keys($fields), $fields));
}
function gpuLog(array $kernels = ['integrateCoreParallelLaunch', 'solveStaticBlock', 'sphereNphase_Kernel'], array $overrides = []): string {
    return implode("\n", array_map(fn($kernel) => launch($kernel, $overrides), $kernels))
        . "\nCuMetal GRB mode: gpu\nSnippetHelloGRB done.\n";
}
function scene(): array {
    return array_merge(labScenes()['rolling'], ['steps'=>2]);
}

test('Single-body capture preserves all 13 state values', function (): void {
    $rows = captureRows();
    $rows[2][8] = -1.25e2;
    $frames = labLoadTsv(captureFile($rows), 1, 1);
    check(count($frames) === 2 && count($frames[0]) === 1 && count($frames[0][0]) === 13, 'Incorrect frame dimensions.');
    near($frames[1][0][0], 1.0);
    near($frames[1][0][7], -125.0);
    near($frames[0][0][6], 1.0);
});
test('Multi-body capture keeps step and body ordering', function (): void {
    $frames = labLoadTsv(captureFile(captureRows(2, 2)), 2, 2);
    check(count($frames) === 3 && count($frames[2]) === 2, 'Incorrect multi-body dimensions.');
    near($frames[2][1][0], 3.0);
});
test('Truncated capture is rejected', function (): void {
    $rows = captureRows();
    array_pop($rows);
    rejects(fn() => labLoadTsv(captureFile($rows), 1, 1), 'missing, duplicate, or reordered');
});
test('Missing native capture is rejected without a PHP warning', function () use ($directory): void {
    set_error_handler(function (int $severity, string $message): never {
        throw new ErrorException($message, 0, $severity);
    });
    try {
        rejects(fn() => labLoadTsv($directory . '/missing.tsv', 1, 1), 'Missing native capture');
    } finally { restore_error_handler(); }
});
test('Unexpected schema is rejected', function (): void {
    $rows = captureRows();
    $rows[0][1] = 'position_x';
    rejects(fn() => labLoadTsv(captureFile($rows), 1, 1), 'unexpected schema');
});
test('Short state and surplus state columns are rejected', function (): void {
    foreach ([-1, 1] as $delta) {
        $rows = captureRows();
        if ($delta < 0) array_pop($rows[1]); else $rows[1][] = 42;
        rejects(fn() => labLoadTsv(captureFile($rows), 1, 1), 'missing, duplicate, or reordered');
    }
});
test('Reordered and duplicate steps are rejected', function (): void {
    $rows = captureRows();
    [$rows[1], $rows[2]] = [$rows[2], $rows[1]];
    rejects(fn() => labLoadTsv(captureFile($rows), 1, 1), 'missing, duplicate, or reordered');
    $rows = captureRows();
    $rows[2][0] = 0;
    rejects(fn() => labLoadTsv(captureFile($rows), 1, 1), 'missing, duplicate, or reordered');
});
test('Reordered and duplicate bodies are rejected', function (): void {
    $rows = captureRows(1, 2);
    [$rows[1], $rows[2]] = [$rows[2], $rows[1]];
    rejects(fn() => labLoadTsv(captureFile($rows), 1, 2), 'missing, duplicate, or reordered');
    $rows = captureRows(1, 2);
    $rows[2][1] = 0;
    rejects(fn() => labLoadTsv(captureFile($rows), 1, 2), 'missing, duplicate, or reordered');
});
test('Extra complete states are rejected', function (): void {
    rejects(fn() => labLoadTsv(captureFile(captureRows(2)), 1, 1), 'extra states');
});
test('NaN, Inf, numeric overflow and nonnumeric data are rejected', function (): void {
    foreach (['NaN', 'Inf', '-Inf', '1e999', '-1e999', 'hello', ''] as $invalid) {
        $rows = captureRows();
        $rows[1][1] = $invalid;
        rejects(fn() => labLoadTsv(captureFile($rows), 1, 1), 'non-finite state');
    }
});
test('Non-unit quaternion is rejected', function (): void {
    $rows = captureRows();
    $rows[1][7] = 2;
    rejects(fn() => labLoadTsv(captureFile($rows), 1, 1), 'Invalid orientation');
});
test('Exact comparison returns zero error', function (): void {
    $frames = [[state()], [state()]];
    $result = labCompare($frames, $frames, 1e-3);
    check($result['comparisonPassed'] === true, 'Comparison did not pass.');
    near($result['maxPositionError'], 0.0);
    near($result['maxScaledError'], 0.0);
});
test('Comparison uses absolute plus reference-scaled relative tolerance', function (): void {
    $cpu = [[state()]];
    $cpu[0][0][0] = 10.0;
    $gpu = $cpu;
    $gpu[0][0][0] += 0.005;
    $gpu[0][0][1] += 0.0005;
    $gpu[0][0][2] -= 0.00025;
    $result = labCompare($cpu, $gpu, 1e-3);
    near($result['maxScaledError'], 0.005 / (1e-5 + 1e-3 * 10));
    near($result['maxPositionError'], sqrt(0.005 ** 2 + 0.0005 ** 2 + 0.00025 ** 2));
});
test('Comparison checks all 13 fields, including angular velocity', function (): void {
    foreach (range(0, 12) as $field) {
        $cpu = [[state()]];
        $gpu = $cpu;
        $gpu[0][0][$field] += 0.1;
        rejects(fn() => labCompare($cpu, $gpu, 1e-3), 'field ' . LAB_FIELDS[$field]);
    }
});
test('Comparison rejects length, body and state-width mismatches', function (): void {
    rejects(fn() => labCompare([], [], 1e-3), 'Capture length mismatch');
    rejects(fn() => labCompare([[state()]], [], 1e-3), 'Capture length mismatch');
    rejects(fn() => labCompare([[state()]], [[state(), state()]], 1e-3), 'Body count mismatch');
    rejects(fn() => labCompare([[state()]], [[array_slice(state(), 0, 12)]], 1e-3), 'State width mismatch');
});
test('Comparison rejects non-finite values on either side', function (): void {
    foreach ([NAN, INF, -INF] as $invalid) {
        $bad = [[state()]];
        $bad[0][0][12] = $invalid;
        rejects(fn() => labCompare([[state()]], $bad, 1e-3), 'Non-finite comparison');
        rejects(fn() => labCompare($bad, [[state()]], 1e-3), 'Non-finite comparison');
    }
});
test('GPU trace counts successful exact Apple metallib launches', function (): void {
    $result = labCheckLog(gpuLog(), scene(), true);
    check($result === ['launches'=>3, 'device'=>'Apple M4'], 'Incorrect trace result.');
});
test('CPU trace completes without claiming GPU execution', function (): void {
    $result = labCheckLog("CuMetal GRB mode: cpu\nSnippetHelloGRB done.\n", scene(), false);
    check($result === ['launches'=>0, 'device'=>null], 'CPU trace claimed GPU execution.');
    rejects(fn() => labCheckLog(str_replace('mode: gpu', 'mode: cpu', gpuLog()), scene(), false), 'non-Apple-GPU work');
});
test('Truncated, wrong-mode and native-error traces are rejected', function (): void {
    foreach ([str_replace('SnippetHelloGRB done.', '', gpuLog()),
        str_replace('mode: gpu', 'mode: cpu', gpuLog()), gpuLog() . 'CuMetal GRB ERROR: test',
        gpuLog() . 'internal error', gpuLog() . 'failed to create compute pipeline', gpuLog() . 'GPU Hang'] as $log) {
        rejects(fn() => labCheckLog($log, scene(), true), 'did not complete successfully');
    }
});
test('Failed, non-GPU, imprecise and non-metallib launches are rejected', function (): void {
    foreach ([['launch_success'=>'false'], ['device'=>'cpu'], ['source'=>'cpu'],
        ['semantic_quality'=>'approximate'], ['device_name'=>'Unknown']] as $fields) {
        rejects(fn() => labCheckLog(gpuLog() . launch('extra', $fields), scene(), true), 'non-Apple-GPU work');
    }
});
test('Missing trace attributes and zero-launch traces are rejected', function (): void {
    foreach (['launch_success="true"', 'device="apple_gpu"', 'source="metallib"',
        'semantic_quality="exact"', 'device_name="Apple M4"'] as $attribute) {
        rejects(fn() => labCheckLog(str_replace($attribute, '', gpuLog()), scene(), true), 'non-Apple-GPU work');
    }
    rejects(fn() => labCheckLog("CuMetal GRB mode: gpu\nSnippetHelloGRB done.\n", scene(), true), 'Missing required GPU kernel');
});
test('Required scene kernels and minimum launch count are enforced', function (): void {
    rejects(fn() => labCheckLog(gpuLog(['integrateCoreParallelLaunch', 'solveStaticBlock']), scene(), true), 'sphereNphase_Kernel');
    rejects(fn() => labCheckLog(gpuLog(), labScenes()['rolling'], true), 'Too few GPU completion records');
    foreach (['mesh', 'boxes'] as $name) {
        $selected = array_merge(labScenes()[$name], ['steps'=>1]);
        rejects(fn() => labCheckLog(gpuLog(), $selected, true), 'Missing required GPU kernel');
    }
});
test('Mesh and box scenes accept their complete required kernel sets', function (): void {
    $kernels = [
        'mesh'=>['integrateCoreParallelLaunch', 'solveStaticBlock', 'midphaseGeneratePairs',
            'sphereTrimeshNarrowphase', 'sortTriangleIndices', 'convexTrimeshPostProcess',
            'convexTrimeshCorrelate', 'convexTrimeshFinishContacts'],
        'boxes'=>['integrateCoreParallelLaunch', 'solveStaticBlock', 'boxBoxNphase_Kernel',
            'ZeroBodies', 'solveBlockPartition', 'writeBackBodies'],
    ];
    foreach ($kernels as $name => $required) {
        $selected = array_merge(labScenes()[$name], ['steps'=>1]);
        $result = labCheckLog(gpuLog($required), $selected, true);
        check($result['launches'] === count($required), 'Incorrect scene launch count.');
        foreach ($required as $missing) {
            rejects(fn() => labCheckLog(gpuLog(array_values(array_diff($required, [$missing]))), $selected, true), $missing);
        }
    }
});
test('Device identity cannot change during a run', function (): void {
    rejects(fn() => labCheckLog(gpuLog() . launch('extra', ['device_name'=>'Apple M3']), scene(), true), 'Device identity changed');
});
test('Subprocess preserves argument boundaries and merges stderr', function () use ($directory): void {
    $path = $directory . '/execute.log';
    $argument = 'literal spaces; $(false)';
    labExecute([PHP_BINARY, '-r', 'echo $argv[1]; fwrite(STDERR, " error-stream");', $argument], getenv(), $path);
    check(file_get_contents($path) === $argument . ' error-stream', 'Shell parsing or output capture changed arguments.');
});
test('Nonzero subprocess status is rejected and its log retained', function () use ($directory): void {
    $path = $directory . '/nonzero.log';
    rejects(fn() => labExecute([PHP_BINARY, '-r', 'echo "diagnostic"; exit(7);'], getenv(), $path), 'exited 7');
    check(file_get_contents($path) === 'diagnostic', 'Failure log was not retained.');
});
test('Fast-exit subprocess cannot bypass the output limit', function () use ($directory): void {
    $path = $directory . '/oversized.log';
    rejects(fn() => labExecute([PHP_BINARY, '-r', 'echo str_repeat("x", 17 * 1024 * 1024);'], getenv(), $path), 'time or output limit');
    clearstatcache(true, $path);
    check(filesize($path) > 16 * 1024 * 1024, 'Oversized child did not exercise the output limit.');
});
test('Timed-out subprocess is terminated promptly', function () use ($directory): void {
    $path = $directory . '/timeout.log';
    $started = microtime(true);
    rejects(fn() => labExecute([PHP_BINARY, '-r', 'echo getmypid(); flush(); sleep(10);'], getenv(), $path, 0.2), 'time or output limit');
    check(microtime(true) - $started < 3, 'Timeout did not bound subprocess lifetime.');
    $pid = (int)file_get_contents($path);
    check($pid > 0, 'Child did not record its PID.');
    if (function_exists('posix_kill')) check(!posix_kill($pid, 0), 'Timed-out child remained alive.');
});
test('Capture directories permit intentional reuse but reject blocked paths', function () use ($directory): void {
    $path = $directory . '/new-output';
    labCreateDirectory($path, true);
    labCreateDirectory($path, true);
    check(is_dir($path), 'Output directory was not created.');
    rejects(fn() => labCreateDirectory($path), 'Cannot create capture directory');
    file_put_contents($directory . '/blocked-directory', 'existing file');
    rejects(fn() => labCreateDirectory($directory . '/blocked-directory', true), 'Cannot create capture directory');
    rejects(fn() => labCreateDirectory($directory . '/missing-parent/child'), 'Cannot create capture directory');
});
test('Successful persistence retains and atomically replaces latest capture', function () use ($directory): void {
    $output = $directory . '/publish-success';
    labCreateDirectory($output);
    labCreateDirectory($output . '/run');
    file_put_contents($output . '/latest.json', '{"id":"previous"}');
    $json = '{"id":"verified","comparisonPassed":true}';
    labPersistCapture($output . '/run', $output, $json);
    check(file_get_contents($output . '/run/result.json') === $json, 'Completed capture was not retained.');
    check(file_get_contents($output . '/latest.json') === $json, 'Latest capture was not replaced.');
    check(glob($output . '/latest-*.tmp') === [], 'Publication left a temporary file.');
});
test('Failed result write preserves the previous latest capture', function () use ($directory): void {
    $output = $directory . '/publish-result-failure';
    labCreateDirectory($output);
    labCreateDirectory($output . '/run');
    labCreateDirectory($output . '/run/result.json');
    $previous = '{"id":"previous"}';
    file_put_contents($output . '/latest.json', $previous);
    rejects(fn() => labPersistCapture($output . '/run', $output, '{"id":"new"}'), 'Cannot write completed capture');
    check(file_get_contents($output . '/latest.json') === $previous, 'Failed result write replaced latest capture.');
    check(glob($output . '/latest-*.tmp') === [], 'Failed result write created a publication temporary.');
});
test('Failed staging write is rejected and retains the completed result', function () use ($directory): void {
    $run = $directory . '/publish-stage-failure';
    labCreateDirectory($run);
    $json = '{"id":"verified"}';
    rejects(fn() => labPersistCapture($run, $run . '/missing-output', $json), 'Cannot stage latest capture');
    check(file_get_contents($run . '/result.json') === $json, 'Staging failure lost the completed result.');
    check(!file_exists($run . '/missing-output/latest.json'), 'Staging failure published latest capture.');
});
test('Failed atomic rename is rejected without damaging the destination', function () use ($directory): void {
    $output = $directory . '/publish-rename-failure';
    labCreateDirectory($output);
    labCreateDirectory($output . '/run');
    labCreateDirectory($output . '/latest.json');
    file_put_contents($output . '/latest.json/keep', 'previous destination');
    $json = '{"id":"verified"}';
    rejects(fn() => labPersistCapture($output . '/run', $output, $json), 'Cannot publish latest capture');
    check(file_get_contents($output . '/latest.json/keep') === 'previous destination', 'Rename failure damaged the destination.');
    check(file_get_contents($output . '/run/result.json') === $json, 'Rename failure lost the completed result.');
    check(glob($output . '/latest-*.tmp') === [], 'Rename failure left a temporary file.');
});

echo "$passed passed; $failed failed. No GPU work was performed.\n";
exit($failed === 0 ? 0 : 1);
