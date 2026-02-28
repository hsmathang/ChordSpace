const { spawn } = require('child_process');

const serverType = process.argv[2] || "notebooklm";
const toolName = process.argv[3];
const toolArgs = process.argv[4] ? JSON.parse(process.argv[4]) : {};

const cmd = "C:\\Program Files\\nodejs\\node.exe";
const args = serverType === "notebooklm"
    ? ["C:\\Users\\Admin\\AppData\\Roaming\\npm\\node_modules\\notebooklm-mcp\\dist\\index.js"]
    : ["C:\\Users\\Admin\\AppData\\Roaming\\npm\\node_modules\\@cmd8\\excalidraw-mcp\\dist\\index.js"];

console.log(`[+] Spawning ${serverType} MCP server natively...`);
const child = spawn(cmd, args);

let initialized = false;

child.stdout.on('data', (data) => {
    const lines = data.toString().split('\n');
    for (const line of lines) {
        if (!line.trim()) continue;
        try {
            const msg = JSON.parse(line);

            // Handle Init Response
            if (msg.id === 1 && msg.result) {
                console.log(`[+] Initialized! Server Version: ${msg.result.serverInfo.version}`);
                child.stdin.write(JSON.stringify({
                    jsonrpc: "2.0",
                    method: "notifications/initialized"
                }) + "\n");

                if (!toolName) {
                    console.log(`[+] Requesting available tools...`);
                    child.stdin.write(JSON.stringify({
                        jsonrpc: "2.0",
                        id: 2,
                        method: "tools/list",
                        params: {}
                    }) + "\n");
                } else {
                    console.log(`[+] Calling tool: ${toolName}...`);
                    child.stdin.write(JSON.stringify({
                        jsonrpc: "2.0",
                        id: 3,
                        method: "tools/call",
                        params: {
                            name: toolName,
                            arguments: toolArgs
                        }
                    }) + "\n");
                }
            }
            // Handle Tool List Response
            else if (msg.id === 2 && msg.result && msg.result.tools) {
                console.log("\n--- AVAILABLE TOOLS ---");
                msg.result.tools.forEach(t => {
                    console.log(`🔸 ${t.name}: ${t.description.substring(0, 80)}...`);
                });
                console.log("------------------------\n");
                process.exit(0);
            }
            // Handle Tool Call Response
            else if (msg.id === 3 && msg.result && msg.result.content) {
                console.log("\n--- TOOL OUTPUT ---");
                msg.result.content.forEach(c => {
                    if (c.type === "text") console.log(c.text);
                    else console.log(JSON.stringify(c, null, 2));
                });
                console.log("-------------------\n");
                process.exit(0);
            }
            else if (msg.error) {
                console.error("[-] MCP Protocol Error:", JSON.stringify(msg.error, null, 2));
                process.exit(1);
            }
        } catch (e) {
            // Ignore non-JSON logs like diagnostic messages or start banners
        }
    }
});

child.stderr.on('data', (data) => {
    // Only print if there's a fatal exception, standard logs can get noisy
    const errStr = data.toString();
    if (errStr.includes("Error") || errStr.includes("Exception")) {
        console.error(`[STDERR] ${errStr}`);
    }
});

// Trigger the initialization sequence
child.stdin.write(JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "antigravity-bypass-js", version: "1.0.0" }
    }
}) + "\n");

// Safety timeout
setTimeout(() => {
    console.error("[-] Timeout: MCP server failed to respond within 15 seconds.");
    process.exit(1);
}, 15000);
