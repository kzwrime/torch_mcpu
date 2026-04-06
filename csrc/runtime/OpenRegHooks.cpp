#include "OpenRegHooks.h"

// LITERALINCLUDE MCPU HOOK REGISTER
namespace c10::mcpu {

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new McpuHooksInterface());

  return true;
}();

} // namespace c10::mcpu
// LITERALINCLUDE MCPU HOOK REGISTER