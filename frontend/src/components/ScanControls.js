import React from 'react';

function ScanControls({
  maxResults,
  setMaxResults,
  importanceCriteria,
  setImportanceCriteria,
  showCategories,
  setShowCategories,
  startDate,
  setStartDate,
  endDate,
  setEndDate,
  onScan,
  isLoading
}) {
  const setDateRange = (days) => {
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - days);
    
    // Format dates as YYYY-MM-DD for the input
    const formatDate = (date) => {
      return date.toISOString().split('T')[0];
    };
    
    setStartDate(formatDate(start));
    setEndDate(formatDate(end));
    // Clear max results when using date filtering
    setMaxResults('');
  };

  const handleMaxResultsChange = (e) => {
    const value = e.target.value;
    // Only set max results if it's a valid number
    if (value && !isNaN(value) && parseInt(value) > 0) {
      setMaxResults(parseInt(value));
      // Clear date range when using max results
      setStartDate('');
      setEndDate('');
    } else {
      setMaxResults('');
    }
  };

  const handleDateChange = (type, value) => {
    if (type === 'start') {
      setStartDate(value);
    } else {
      setEndDate(value);
    }
    // Clear max results when using date filtering
    setMaxResults('');
  };

  const clearDates = () => {
    setStartDate('');
    setEndDate('');
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <input
              type="number"
              value={maxResults}
              onChange={handleMaxResultsChange}
              min="1"
              max="50"
              className="border rounded px-3 py-2 w-24"
              placeholder="Number of emails"
              disabled={startDate || endDate}
            />
            {(startDate || endDate) && (
              <div className="absolute -top-6 left-0 text-xs text-gray-500">
                Disabled when using date filter
              </div>
            )}
          </div>
          <button
            onClick={onScan}
            disabled={isLoading || (!maxResults && !startDate && !endDate)}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {isLoading ? 'Scanning...' : 'Scan Emails'}
          </button>
        </div>
        <div className="flex items-center space-x-3">
          <span className="text-sm font-medium text-gray-700">Show Categories</span>
          <button
            type="button"
            onClick={() => setShowCategories(!showCategories)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none cursor-pointer ${
              showCategories ? 'bg-blue-600' : 'bg-gray-200'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                showCategories ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-gray-700">Quick Select:</span>
          <button
            onClick={() => setDateRange(1)}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Last 24 hours
          </button>
          <button
            onClick={() => setDateRange(7)}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Last 7 days
          </button>
          <button
            onClick={() => setDateRange(30)}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Last 30 days
          </button>
          <button
            onClick={clearDates}
            className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
          >
            Clear dates
          </button>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label
              htmlFor="startDate"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Start Date
            </label>
            <input
              type="date"
              id="startDate"
              value={startDate}
              onChange={(e) => handleDateChange('start', e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label
              htmlFor="endDate"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              End Date
            </label>
            <input
              type="date"
              id="endDate"
              value={endDate}
              onChange={(e) => handleDateChange('end', e.target.value)}
              className="w-full border rounded px-3 py-2"
            />
          </div>
        </div>
      </div>

      <div>
        <label
          htmlFor="importancePrompt"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          What makes an email important?
        </label>
        <textarea
          id="importancePrompt"
          value={importanceCriteria}
          onChange={(e) => setImportanceCriteria(e.target.value)}
          rows="3"
          className="w-full border rounded px-3 py-2"
          placeholder="Describe what makes an email important to you..."
        />
      </div>
    </div>
  );
}

export default ScanControls; 