#pragma once
#include <string>

namespace elb {
// Transport-independent NDJSON envelope. Implementations must correlate request_id,
// enforce a deadline and return a complete result/error envelope. No Qt dependency.
class ILlmTransport {
public:
    virtual ~ILlmTransport() = default;
    virtual std::string exchange(const std::string & request_json,
                                 int timeout_ms) = 0;
};
} // namespace elb
