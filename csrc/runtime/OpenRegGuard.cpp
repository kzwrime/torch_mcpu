#include "OpenRegGuard.h"

namespace c10::mcpu {

// LITERALINCLUDE MCPU GUARD REGISTRATION
C10_REGISTER_GUARD_IMPL(PrivateUse1, McpuGuardImpl);
// LITERALINCLUDE MCPU GUARD REGISTRATION

} // namespace c10::mcpu
