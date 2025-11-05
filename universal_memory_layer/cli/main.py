"""Main CLI interface for the Universal Memory Layer."""

import argparse
import sys
import os
import signal
from typing import Optional, List, Dict, Any

from ..models.config import MemoryConfig
from ..conversation_manager import ConversationManager, ConversationManagerError
from ..logging_config import get_logger, setup_logging
from ..errors import (
    UMLError, 
    ConversationError, 
    handle_errors, 
    safe_execute,
    create_user_friendly_message
)

logger = get_logger(__name__)


class CLIError(UMLError):
    """Exception raised for CLI-specific errors."""
    pass


class UniversalMemoryCLI:
    """
    Command-line interface for the Universal Memory Layer.
    
    Provides interactive chat functionality with model selection,
    session management, and graceful exit handling.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize CLI with configuration.
        
        Args:
            config: Memory configuration. If None, uses default configuration.
        """
        self.config = config or MemoryConfig()
        self.conversation_manager: Optional[ConversationManager] = None
        self.current_session: Optional[Dict[str, Any]] = None
        self.running = False
        
        # Set up signal handlers for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Configure logging for CLI
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for CLI interface."""
        # Set up UML logging system
        log_level = os.getenv("UML_LOG_LEVEL", "WARNING").upper()
        
        try:
            setup_logging(
                log_level=log_level,
                console_output=False,  # Disable console output for CLI to avoid interference
                file_output=True
            )
            logger.debug("CLI logging configured")
        except Exception as e:
            # Fallback to basic logging if UML logging fails
            import logging
            logging.basicConfig(
                level=getattr(logging, log_level, logging.WARNING),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            logger.warning(f"Failed to set up UML logging, using basic logging: {e}")
        
        # Reduce noise from external libraries
        import logging
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle interrupt signals for graceful shutdown."""
        print("\n\nReceived interrupt signal. Shutting down gracefully...")
        self.running = False
        sys.exit(0)
    
    @handle_errors(
        error_mapping={
            ConversationManagerError: CLIError,
            Exception: CLIError
        }
    )
    def initialize(self) -> None:
        """
        Initialize the CLI and conversation manager.
        
        Raises:
            CLIError: If initialization fails
        """
        print("Initializing Universal Memory Layer...")
        self.conversation_manager = ConversationManager(self.config)
        self.conversation_manager.initialize()
        print("✓ Initialization complete")
        logger.info("CLI initialized successfully")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
            
        Raises:
            CLIError: If conversation manager is not initialized
        """
        if not self.conversation_manager:
            raise CLIError("Conversation manager not initialized")
        
        return self.conversation_manager.get_available_models()
    
    def display_model_menu(self) -> None:
        """Display available models in a formatted menu."""
        try:
            models = self.get_available_models()
            
            if not models:
                print("\n❌ No models available!")
                print("Please ensure you have set the appropriate API keys:")
                print("  - OPENAI_API_KEY for ChatGPT/GPT-4")
                print("  - ANTHROPIC_API_KEY for Claude")
                print("  - GOOGLE_API_KEY for Gemini")
                return
            
            print("\n📋 Available Models:")
            print("=" * 40)
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            print("=" * 40)
            
        except CLIError:
            raise
        except Exception as e:
            raise CLIError(f"Failed to display model menu: {e}")
    
    def select_model(self) -> Optional[str]:
        """
        Interactive model selection.
        
        Returns:
            Selected model name or None if cancelled
            
        Raises:
            CLIError: If model selection fails
        """
        try:
            models = self.get_available_models()
            
            if not models:
                self.display_model_menu()
                return None
            
            self.display_model_menu()
            
            while True:
                try:
                    choice = input(f"\nSelect a model (1-{len(models)}) or 'q' to quit: ").strip()
                    
                    if choice.lower() in ['q', 'quit', 'exit']:
                        return None
                    
                    model_index = int(choice) - 1
                    if 0 <= model_index < len(models):
                        selected_model = models[model_index]
                        print(f"✓ Selected model: {selected_model}")
                        return selected_model
                    else:
                        print(f"❌ Invalid choice. Please enter a number between 1 and {len(models)}")
                
                except ValueError:
                    print("❌ Invalid input. Please enter a number or 'q' to quit")
                except KeyboardInterrupt:
                    print("\n\nOperation cancelled.")
                    return None
                
        except CLIError:
            raise
        except Exception as e:
            raise CLIError(f"Model selection failed: {e}")
    
    def start_session(self, model: Optional[str] = None) -> bool:
        """
        Start a new conversation session.
        
        Args:
            model: Model to use. If None, prompts for selection.
            
        Returns:
            True if session started successfully, False otherwise
            
        Raises:
            CLIError: If session start fails
        """
        try:
            if not self.conversation_manager:
                raise CLIError("Conversation manager not initialized")
            
            # Select model if not provided
            if not model:
                model = self.select_model()
                if not model:
                    return False
            
            # Start conversation session
            self.current_session = self.conversation_manager.start_conversation(model=model)
            
            print(f"\n🚀 Started conversation with {model}")
            print(f"📊 Memory contains {self.current_session['memory_stats']['total_messages']} messages")
            
            return True
            
        except ConversationManagerError as e:
            raise CLIError(f"Failed to start session: {e}")
        except Exception as e:
            raise CLIError(f"Unexpected error starting session: {e}")
    
    def display_help(self) -> None:
        """Display help information for CLI commands."""
        help_text = """
🔧 Universal Memory Layer - Commands:

Chat Commands:
  <message>          Send a message to the current model
  
Model Commands:
  /switch            Switch to a different model
  /models            Show available models
  /current           Show current model
  
Memory Commands:
  /recent [n]        Show last n messages (default: 5)
  /search <query>    Search for relevant messages
  /stats             Show memory statistics
  /clear             Clear all memory (requires confirmation)
  
Utility Commands:
  /help              Show this help message
  /quit, /exit       Exit the CLI
  
Examples:
  Hello, how are you?
  /switch
  /recent 10
  /search "machine learning"
"""
        print(help_text)
    
    def display_welcome(self) -> None:
        """Display welcome message and basic instructions."""
        welcome_text = """
🌟 Welcome to Universal Memory Layer CLI!

This tool allows you to chat with different AI models while maintaining
conversation history and context across model switches.

Type '/help' for available commands or just start chatting!
Type '/quit' or press Ctrl+C to exit.
"""
        print(welcome_text)
    
    def display_recent_messages(self, count: int = 5) -> None:
        """
        Display recent messages from conversation history.
        
        Args:
            count: Number of recent messages to display
        """
        try:
            if not self.conversation_manager:
                print("❌ Conversation manager not initialized")
                return
            
            messages = self.conversation_manager.get_context("", strategy="recent", k=count)
            
            if not messages:
                print("📭 No messages in conversation history")
                return
            
            print(f"\n📜 Last {len(messages)} messages:")
            print("=" * 50)
            
            for msg in messages:
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                role_icon = "👤" if msg.role == "user" else "🤖"
                model_info = f"({msg.model})" if msg.role == "assistant" else ""
                
                print(f"{role_icon} [{timestamp}] {msg.role.title()} {model_info}")
                print(f"   {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
                print()
            
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Failed to display recent messages: {e}")
    
    def search_messages(self, query: str, count: int = 5) -> None:
        """
        Search for relevant messages in conversation history.
        
        Args:
            query: Search query
            count: Number of results to display
        """
        try:
            if not self.conversation_manager:
                print("❌ Conversation manager not initialized")
                return
            
            if not query.strip():
                print("❌ Please provide a search query")
                return
            
            messages = self.conversation_manager.get_context(query, strategy="relevant", k=count)
            
            if not messages:
                print(f"🔍 No relevant messages found for: '{query}'")
                return
            
            print(f"\n🔍 Search results for '{query}' ({len(messages)} found):")
            print("=" * 50)
            
            for msg in messages:
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                role_icon = "👤" if msg.role == "user" else "🤖"
                model_info = f"({msg.model})" if msg.role == "assistant" else ""
                
                print(f"{role_icon} [{timestamp}] {msg.role.title()} {model_info}")
                print(f"   {msg.content[:150]}{'...' if len(msg.content) > 150 else ''}")
                print()
            
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Failed to search messages: {e}")
    
    def display_memory_stats(self) -> None:
        """Display memory statistics."""
        try:
            if not self.conversation_manager:
                print("❌ Conversation manager not initialized")
                return
            
            stats = self.conversation_manager.get_memory_stats()
            
            print("\n📊 Memory Statistics:")
            print("=" * 30)
            print(f"Total messages: {stats.get('total_messages', 0)}")
            print(f"User messages: {stats.get('user_messages', 0)}")
            print(f"Assistant messages: {stats.get('assistant_messages', 0)}")
            print(f"Available models: {', '.join(stats.get('available_models', []))}")
            print(f"Current model: {stats.get('current_model', 'None')}")
            print("=" * 30)
            
        except Exception as e:
            print(f"❌ Failed to display memory stats: {e}")
    
    def clear_memory_interactive(self) -> None:
        """Interactive memory clearing with confirmation."""
        try:
            if not self.conversation_manager:
                print("❌ Conversation manager not initialized")
                return
            
            print("\n⚠️  WARNING: This will permanently delete all conversation history!")
            confirmation = input("Type 'DELETE' to confirm: ").strip()
            
            if confirmation == "DELETE":
                self.conversation_manager.clear_memory(confirm=True)
                print("✅ Memory cleared successfully")
            else:
                print("❌ Memory clear cancelled")
                
        except Exception as e:
            print(f"❌ Failed to clear memory: {e}")
    
    def switch_model_interactive(self) -> bool:
        """
        Interactive model switching.
        
        Returns:
            True if model was switched, False otherwise
        """
        try:
            if not self.conversation_manager:
                print("❌ Conversation manager not initialized")
                return False
            
            current_model = self.conversation_manager.get_current_model()
            print(f"\nCurrent model: {current_model}")
            
            new_model = self.select_model()
            if not new_model:
                return False
            
            if new_model == current_model:
                print("✓ Already using that model")
                return False
            
            self.conversation_manager.switch_model(new_model)
            print(f"✅ Switched to {new_model}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to switch model: {e}")
            return False
    
    def process_command(self, user_input: str) -> bool:
        """
        Process CLI commands.
        
        Args:
            user_input: User input string
            
        Returns:
            True to continue chat loop, False to exit
        """
        command_parts = user_input.strip().split()
        command = command_parts[0].lower()
        
        try:
            if command in ["/quit", "/exit"]:
                return False
            
            elif command == "/help":
                self.display_help()
            
            elif command == "/switch":
                self.switch_model_interactive()
            
            elif command == "/models":
                self.display_model_menu()
            
            elif command == "/current":
                current = self.conversation_manager.get_current_model() if self.conversation_manager else None
                print(f"Current model: {current or 'None'}")
            
            elif command == "/recent":
                count = 5  # default
                if len(command_parts) > 1:
                    try:
                        count = int(command_parts[1])
                    except ValueError:
                        print("❌ Invalid number for recent count")
                        return True
                self.display_recent_messages(count)
            
            elif command == "/search":
                if len(command_parts) < 2:
                    print("❌ Usage: /search <query>")
                else:
                    query = " ".join(command_parts[1:])
                    self.search_messages(query)
            
            elif command == "/stats":
                self.display_memory_stats()
            
            elif command == "/clear":
                self.clear_memory_interactive()
            
            else:
                print(f"❌ Unknown command: {command}")
                print("Type '/help' for available commands")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing command: {e}")
            return True
    
    def send_message(self, message: str) -> None:
        """
        Send a message to the current model and display response.
        
        Args:
            message: User message to send
        """
        try:
            if not self.conversation_manager:
                print("❌ Conversation manager not initialized")
                return
            
            current_model = self.conversation_manager.get_current_model()
            if not current_model:
                print("❌ No model selected")
                return
            
            print(f"\n🤖 {current_model} is thinking...")
            
            # Send message and get response
            response = self.conversation_manager.chat(message)
            
            # Display response
            print(f"\n🤖 {current_model}:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except ConversationManagerError as e:
            print(f"❌ Conversation error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    def chat_loop(self) -> None:
        """Main interactive chat loop."""
        try:
            while self.running:
                try:
                    # Get user input
                    current_model = self.conversation_manager.get_current_model() if self.conversation_manager else "None"
                    prompt = f"\n[{current_model}] You: "
                    user_input = input(prompt).strip()
                    
                    # Skip empty input
                    if not user_input:
                        continue
                    
                    # Process commands
                    if user_input.startswith("/"):
                        if not self.process_command(user_input):
                            break  # Exit command received
                    else:
                        # Send regular message
                        self.send_message(user_input)
                
                except KeyboardInterrupt:
                    print("\n\nUse '/quit' to exit or Ctrl+C again to force quit.")
                    try:
                        # Give user a chance to use /quit
                        input("Press Enter to continue or Ctrl+C again to force quit...")
                    except KeyboardInterrupt:
                        print("\nForce quit detected. Goodbye! 👋")
                        break
                except EOFError:
                    print("\nEnd of input detected. Goodbye! 👋")
                    break
                
        except Exception as e:
            print(f"❌ Chat loop error: {e}")
        finally:
            self.running = False
    
    def run(self) -> int:
        """
        Run the main CLI application.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            # Initialize the system
            self.initialize()
            
            # Display welcome message
            self.display_welcome()
            
            # Start a session
            if not self.start_session():
                print("No session started. Exiting.")
                return 0
            
            # Set running flag
            self.running = True
            
            print("\n" + "=" * 60)
            print("💬 Chat started! Type your message or '/help' for commands.")
            print("=" * 60)
            
            # Start the interactive chat loop
            self.chat_loop()
            
            print("\nGoodbye! 👋")
            return 0
            
        except CLIError as e:
            print(f"❌ CLI Error: {e}")
            return 1
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            return 0
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for CLI.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Universal Memory Layer - Chat with AI models while maintaining conversation history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start interactive CLI
  %(prog)s --model chatgpt    # Start with specific model
  %(prog)s --help             # Show this help message

Environment Variables:
  OPENAI_API_KEY              # OpenAI API key for ChatGPT/GPT-4
  ANTHROPIC_API_KEY           # Anthropic API key for Claude
  GOOGLE_API_KEY              # Google API key for Gemini
  UML_DATABASE_PATH           # Path to SQLite database (default: memory.db)
  UML_EMBEDDING_PROVIDER      # Embedding provider: openai or huggingface
  UML_LOG_LEVEL               # Logging level: DEBUG, INFO, WARNING, ERROR
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model to use for conversation (chatgpt, gpt-4, claude, gemini)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (not implemented yet)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Universal Memory Layer CLI v1.0.0"
    )
    
    return parser


def main() -> int:
    """
    Main entry point for the CLI application.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Create and run CLI
        cli = UniversalMemoryCLI()
        return cli.run()
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())