-- wrk Lua script for heart disease prediction testing
-- This script sends POST requests with JSON data

-- Read sample data from file
local file = io.open("sample_for_wrk.json", "r")
local sample_data = file:read("*all")
file:close()

-- Set up the request
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = sample_data

-- Optional: Print response for debugging
function response(status, headers, body)
    if status ~= 200 then
        print("Error response: " .. status)
        print("Body: " .. body)
    end
end

-- Track request statistics  
function done(summary, latency, requests)
    print("\n🏁 WRK PERFORMANCE TEST RESULTS:")
    print("================================")
    print(string.format("Requests: %d", summary.requests))
    print(string.format("Duration: %.2fs", summary.duration / 1000000))
    print(string.format("Errors: %d", summary.errors.connect + summary.errors.read + summary.errors.write + summary.errors.status + summary.errors.timeout))
    print(string.format("Requests/sec: %.2f", summary.requests / (summary.duration / 1000000)))
    print(string.format("Avg Latency: %.2fms", latency.mean / 1000))
    print(string.format("Max Latency: %.2fms", latency.max / 1000))
    print(string.format("99th Percentile: %.2fms", latency:percentile(99) / 1000))
end
