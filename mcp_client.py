import asyncio
import json
import argparse
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run(server_name, command, args_list, tool_name, tool_args_json):
    server_params = StdioServerParameters(
        command=command,
        args=args_list,
        env=None
    )
    print(f"Starting {server_name} MCP server via {command}...")
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"[{server_name}] Initialized successfully.")
                
                # List available tools if none specified
                if not tool_name:
                    tools = await session.list_tools()
                    print(f"\n[{server_name}] Available tools:")
                    for t in tools.tools:
                        print(f"  - {t.name}: {t.description}")
                    return

                # Call the specific tool
                parsed_args = json.loads(tool_args_json) if tool_args_json else {}
                print(f"\n[{server_name}] Calling tool '{tool_name}' with args {parsed_args}...")
                
                result = await session.call_tool(tool_name, arguments=parsed_args)
                
                print(f"\n[{server_name}] Output:")
                for content in result.content:
                    if content.type == "text":
                        print(content.text)
                    else:
                        print(content)
                        
    except Exception as e:
        print(f"Error communicating with MCP server: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal MCP Client for Antigravity Bypass")
    parser.add_argument("server", choices=["notebooklm", "excalidraw"], help="Target MCP server")
    parser.add_argument("--tool", help="Name of the tool to execute")
    parser.add_argument("--args", default="{}", help="JSON string representing tool arguments")
    
    args = parser.parse_args()
    
    cmd = "C:\\Program Files\\nodejs\\node.exe"
    if args.server == "notebooklm":
        cmd_args = ["C:\\Users\\Admin\\AppData\\Roaming\\npm\\node_modules\\notebooklm-mcp\\dist\\index.js"]
    elif args.server == "excalidraw":
        cmd_args = ["C:\\Users\\Admin\\AppData\\Roaming\\npm\\node_modules\\@cmd8\\excalidraw-mcp\\dist\\index.js"]

    asyncio.run(run(args.server, cmd, cmd_args, args.tool, args.args))
