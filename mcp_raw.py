import subprocess
import json
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python mcp_raw.py [notebooklm|excalidraw] [tool_name] [args_json]")
        return
        
    server_type = sys.argv[1]
    
    cmd = "C:\\Program Files\\nodejs\\node.exe"
    if server_type == "notebooklm":
        args = ["C:\\Users\\Admin\\AppData\\Roaming\\npm\\node_modules\\notebooklm-mcp\\dist\\index.js"]
    elif server_type == "excalidraw":
        args = ["C:\\Users\\Admin\\AppData\\Roaming\\npm\\node_modules\\@cmd8\\excalidraw-mcp\\dist\\index.js"]
    else:
        print(f"Unknown server: {server_type}")
        return
        
    print(f"Spawning {server_type} server...", flush=True)
    proc = subprocess.Popen([cmd] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 1. Initialize
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "raw-client", "version": "1.0.0"}
        }
    }
    
    proc.stdin.write(json.dumps(init_req) + "\n")
    proc.stdin.flush()
    
    resp_init = json.loads(proc.stdout.readline())
    print(f"Initialized (Server version: {resp_init.get('result', {}).get('serverInfo', {}).get('version')})")
    
    # 2. Sent initialized notification
    proc.stdin.write(json.dumps({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }) + "\n")
    proc.stdin.flush()

    if len(sys.argv) == 2:
        # Just list tools
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }) + "\n")
        proc.stdin.flush()
        
        resp_tools = json.loads(proc.stdout.readline())
        tools = resp_tools.get("result", {}).get("tools", [])
        print("\nAvailable Tools:")
        for t in tools:
            print(f" - {t['name']}: {t['description']}")
            
    elif len(sys.argv) >= 3:
        tool_name = sys.argv[2]
        tool_args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args
            }
        }) + "\n")
        proc.stdin.flush()
        
        print(f"\nCalling {tool_name}...")
        resp_call = json.loads(proc.stdout.readline())
        
        content = resp_call.get("result", {}).get("content", [])
        for block in content:
            if block.get("type") == "text":
                print(block.get("text"))
            else:
                print(json.dumps(block, indent=2))
                
    # Terminate gracefully
    proc.terminate()

if __name__ == "__main__":
    main()
