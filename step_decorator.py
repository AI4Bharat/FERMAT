from typing import Any, Dict
import json
class Step:

    _state = {}

    def __init__(self, id: str):
        self.id = id
    

    def __call__(self, original_class: type) -> type:
        class Wrapper(original_class):
            _step = self

            def get_id(self) -> str:
                return self._step.id

            def set_state(self, key: str, value: Any) -> None:
                self._step._state[key] = value

            def get_state(self, key: str) -> Any:
                return self._step._state.get(key, None)

            def has_state(self, key: str) -> bool:
                return key in self._step._state
            
            ## Output State
            def output_state(self) -> Dict[str, Any]:
                return self._step._state
            
            def put_state(self, state: Dict[str, Any]) -> None:
                self._step._state = state


        return Wrapper
