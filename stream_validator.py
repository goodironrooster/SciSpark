# stream_validator.py
from buffer_manager import BufferManager
from stream_status import StreamStatus, StreamValidationType

class StreamValidator:
    def __init__(self, buffer_manager: BufferManager):
        self.buffer_manager = buffer_manager

    def validate_stream(self, data: bytes, validation_type: StreamValidationType) -> StreamStatus:
        try:
            # Write data to buffer for validation
            buffer_status = self.buffer_manager.write(data)
            if not buffer_status.success:
                return StreamStatus(
                    success=False,
                    validation_type=validation_type,
                    error="Buffer write failed",
                    is_valid=False
                )
            
            # Perform validation based on type
            if validation_type == StreamValidationType.ENCODING:
                try:
                    # Try to decode as UTF-8
                    data.decode('utf-8')
                    is_valid = True
                except UnicodeError:
                    return StreamStatus(
                        success=False,
                        validation_type=validation_type,
                        error="Invalid UTF-8 encoding",
                        is_valid=False
                    )
            
            elif validation_type == StreamValidationType.INTEGRITY:
                # Check data integrity
                peek_status, peek_data = self.buffer_manager.peek()
                is_valid = peek_status.success and peek_data == data
            
            elif validation_type == StreamValidationType.BOUNDARY:
                # Check buffer boundaries
                is_valid = buffer_status.current_usage <= self.buffer_manager.max_size
            
            elif validation_type == StreamValidationType.CORRUPTION:
                # Simple corruption check
                peek_status, peek_data = self.buffer_manager.peek()
                if not peek_status.success:
                    return StreamStatus(
                        success=False,
                        validation_type=validation_type,
                        error="Failed to peek buffer data",
                        is_valid=False
                    )
                is_valid = len(peek_data) > 0 and all(b != 0 for b in peek_data)
            
            elif validation_type == StreamValidationType.FULL:
                try:
                    # Encoding check
                    data.decode('utf-8')
                    
                    # Integrity check
                    peek_status, peek_data = self.buffer_manager.peek()
                    if not peek_status.success:
                        return StreamStatus(
                            success=False,
                            validation_type=validation_type,
                            error="Failed to peek buffer data",
                            is_valid=False
                        )
                    
                    # Combined validation
                    is_valid = (
                        peek_status.success and 
                        peek_data == data and
                        buffer_status.current_usage <= self.buffer_manager.max_size and
                        len(peek_data) > 0 and
                        all(b != 0 for b in peek_data)
                    )
                except UnicodeError:
                    return StreamStatus(
                        success=False,
                        validation_type=validation_type,
                        error="Invalid UTF-8 encoding in full validation",
                        is_valid=False
                    )
            
            else:
                return StreamStatus(
                    success=False,
                    validation_type=validation_type,
                    error="Unknown validation type",
                    is_valid=False
                )

            # Success case
            return StreamStatus(
                success=True,
                validation_type=validation_type,
                data=data,
                position=buffer_status.current_usage,
                is_valid=is_valid
            )

        except Exception as e:
            return StreamStatus(
                success=False,
                validation_type=validation_type,
                error=str(e),
                is_valid=False
            )

    def validate_content(self, content: str, expected_format: str = "text") -> bool:
        """Legacy method for basic content validation"""
        try:
            if not content:
                return False
                
            if expected_format == "text":
                # Basic text validation
                if not isinstance(content, str) or not content.strip():
                    return False
                
                # Try to encode and decode to check for valid characters
                try:
                    encoded = content.encode('utf-8')
                    encoded.decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    return False
                
                # Check for minimum content length (adjustable)
                if len(content.strip()) < 1:
                    return False
                
                return True
            
            elif expected_format == "binary":
                # For binary data validation
                if not isinstance(content, (bytes, bytearray)):
                    try:
                        # Try to encode if it's a string
                        content.encode('utf-8')
                    except:
                        return False
                return True
            
            elif expected_format == "json":
                # For JSON validation
                import json
                try:
                    json.loads(content)
                    return True
                except:
                    return False
            
            return True  # Default case for unknown formats
            
        except Exception:
            return False

    def check_stream_integrity(self, data: bytes) -> StreamStatus:
        """Additional method to check stream integrity"""
        try:
            # Write to buffer
            write_status = self.buffer_manager.write(data)
            if not write_status.success:
                return StreamStatus(
                    success=False,
                    validation_type=StreamValidationType.INTEGRITY,
                    error="Failed to write to buffer",
                    is_valid=False
                )
            
            # Read back and compare
            read_status, read_data = self.buffer_manager.read()
            if not read_status.success:
                return StreamStatus(
                    success=False,
                    validation_type=StreamValidationType.INTEGRITY,
                    error="Failed to read from buffer",
                    is_valid=False
                )
            
            # Compare data
            is_valid = data == read_data
            
            return StreamStatus(
                success=True,
                validation_type=StreamValidationType.INTEGRITY,
                data=read_data,
                position=write_status.current_usage,
                is_valid=is_valid
            )
            
        except Exception as e:
            return StreamStatus(
                success=False,
                validation_type=StreamValidationType.INTEGRITY,
                error=str(e),
                is_valid=False
            )

    def validate_stream_sequence(self, data: bytes) -> StreamStatus:
        """Validate a sequence of stream operations"""
        try:
            # Clear buffer first
            clear_status = self.buffer_manager.clear()
            if not clear_status.success:
                return StreamStatus(
                    success=False,
                    validation_type=StreamValidationType.FULL,
                    error="Failed to clear buffer",
                    is_valid=False
                )
            
            # Write data
            write_status = self.buffer_manager.write(data)
            if not write_status.success:
                return StreamStatus(
                    success=False,
                    validation_type=StreamValidationType.FULL,
                    error="Failed to write data",
                    is_valid=False
                )
            
            # Peek data
            peek_status, peek_data = self.buffer_manager.peek()
            if not peek_status.success:
                return StreamStatus(
                    success=False,
                    validation_type=StreamValidationType.FULL,
                    error="Failed to peek data",
                    is_valid=False
                )
            
            # Read data
            read_status, read_data = self.buffer_manager.read()
            if not read_status.success:
                return StreamStatus(
                    success=False,
                    validation_type=StreamValidationType.FULL,
                    error="Failed to read data",
                    is_valid=False
                )
            
            # Validate sequence
            is_valid = (
                data == peek_data and
                data == read_data and
                write_status.current_usage == len(data) and
                read_status.current_usage == 0  # Should be empty after read
            )
            
            return StreamStatus(
                success=True,
                validation_type=StreamValidationType.FULL,
                data=read_data,
                position=read_status.current_usage,
                is_valid=is_valid
            )
            
        except Exception as e:
            return StreamStatus(
                success=False,
                validation_type=StreamValidationType.FULL,
                error=str(e),
                is_valid=False
            )
