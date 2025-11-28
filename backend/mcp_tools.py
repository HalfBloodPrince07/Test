# backend/mcp_tools.py - Enhanced Model Context Protocol Tools

from typing import Dict, Any, List, Callable, Optional
from loguru import logger
from pydantic import BaseModel, Field
import inspect
import json


class ToolParameter(BaseModel):
    """Schema for a tool parameter"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class ToolSchema(BaseModel):
    """Schema for a tool"""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    returns: str = "Any"


class MCPToolRegistry:
    """
    Enhanced registry for MCP (Model Context Protocol) tools.
    
    Provides:
    - Tool registration with schema validation
    - Automatic schema generation from function signatures
    - Tool execution with parameter validation
    - Tool discovery for AI agents
    """
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, ToolSchema] = {}
        logger.info("Enhanced MCP Tool Registry initialized")
    
    def register_tool(
        self, 
        name: str, 
        handler: Callable,
        description: Optional[str] = None,
        schema: Optional[ToolSchema] = None
    ):
        """
        Register a tool with the registry.
        
        Args:
            name: Tool name
            handler: Async function to execute
            description: Tool description (auto-generated from docstring if not provided)
            schema: Tool schema (auto-generated from signature if not provided)
        """
        self.tools[name] = handler
        
        # Generate schema if not provided
        if schema:
            self.schemas[name] = schema
        else:
            self.schemas[name] = self._generate_schema(name, handler, description)
        
        logger.info(f"✅ Registered MCP tool: {name}")
    
    def _generate_schema(
        self, 
        name: str, 
        handler: Callable,
        description: Optional[str] = None
    ) -> ToolSchema:
        """Auto-generate schema from function signature"""
        
        # Get description from docstring
        desc = description or handler.__doc__ or f"Execute {name} tool"
        desc = desc.strip().split('\n')[0]  # First line only
        
        # Get parameters from signature
        sig = inspect.signature(handler)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'request', 'background_tasks']:
                continue
            
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_name = getattr(param.annotation, '__name__', str(param.annotation))
                param_type = type_name.lower()
            
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parameter: {param_name}",
                required=required,
                default=default
            ))
        
        return ToolSchema(
            name=name,
            description=desc,
            parameters=parameters
        )
    
    async def execute(
        self, 
        name: str, 
        params: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool with the given parameters.
        
        Args:
            name: Tool name
            params: Tool parameters
            
        Returns:
            Tool execution result
        """
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        
        handler = self.tools[name]
        schema = self.schemas.get(name)
        
        # Validate required parameters
        if schema:
            for param in schema.parameters:
                if param.required and param.name not in params:
                    raise ValueError(f"Missing required parameter: {param.name}")
        
        # Execute
        try:
            if inspect.iscoroutinefunction(handler):
                return await handler(**params)
            else:
                return handler(**params)
        except TypeError as e:
            # Try calling with params dict directly
            if 'params' in inspect.signature(handler).parameters:
                if inspect.iscoroutinefunction(handler):
                    return await handler(params=params)
                else:
                    return handler(params=params)
            raise
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with their schemas"""
        return [
            {
                "name": name,
                "description": schema.description,
                "parameters": [p.dict() for p in schema.parameters]
            }
            for name, schema in self.schemas.items()
        ]
    
    def get_tool_schema(self, name: str) -> Dict[str, Any]:
        """
        Get tool schema in OpenAPI/JSON Schema format.
        Compatible with function calling APIs.
        """
        if name not in self.schemas:
            raise ValueError(f"Tool not found: {name}")
        
        schema = self.schemas[name]
        
        # Convert to JSON Schema format
        properties = {}
        required = []
        
        for param in schema.parameters:
            properties[param.name] = {
                "type": self._convert_type(param.type),
                "description": param.description
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        return {
            "name": name,
            "description": schema.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def _convert_type(self, type_str: str) -> str:
        """Convert Python type to JSON Schema type"""
        type_map = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "dict": "object",
            "any": "string"
        }
        return type_map.get(type_str.lower(), "string")
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get tools in OpenAI function calling format.
        Useful for integration with OpenAI-compatible APIs.
        """
        return [
            {
                "type": "function",
                "function": self.get_tool_schema(name)
            }
            for name in self.tools
        ]
    
    def get_mcp_manifest(self) -> Dict[str, Any]:
        """
        Get MCP manifest for tool discovery.
        """
        return {
            "name": "locallens-tools",
            "version": "2.0.0",
            "description": "LocalLens document search and indexing tools",
            "tools": self.list_tools()
        }


# Pre-defined tool decorators for easy registration
def mcp_tool(registry: MCPToolRegistry, name: str = None, description: str = None):
    """
    Decorator for registering MCP tools.
    
    Usage:
        @mcp_tool(registry, "search_documents", "Search through indexed documents")
        async def search(query: str, top_k: int = 5):
            ...
    """
    def decorator(func: Callable):
        tool_name = name or func.__name__
        registry.register_tool(tool_name, func, description)
        return func
    return decorator
