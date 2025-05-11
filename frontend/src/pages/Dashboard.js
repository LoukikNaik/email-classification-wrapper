import React, { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import EmailList from '../components/EmailList';
import ScanControls from '../components/ScanControls';
import { scanEmails, clearCache } from '../utils/api';

const DEFAULT_IMPORTANCE_CRITERIA = `An email is Important if it is related to:

Job Opportunities: including interview invitations, assessment notices, job rejections or offers. No userless emails like Linkedin asking you to apply for jobs, etc

Housing: including rent payments, building notifications, housing assistance programs, or tenant communications.

Banking: specifically transaction alerts, balance updates, or account activity.
Emails that are promotional, newsletters, marketing content, or unrelated to these categories are Useless.`;

function Dashboard() {
  const { logout } = useAuth();
  const [maxResults, setMaxResults] = useState(10);
  const [importanceCriteria, setImportanceCriteria] = useState(DEFAULT_IMPORTANCE_CRITERIA);
  const [showCategories, setShowCategories] = useState(false);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [emails, setEmails] = useState({ important: [], nonImportant: [] });

  const handleScan = async () => {
    setIsLoading(true);
    setError('');
    try {
      const results = await scanEmails({
        maxResults,
        importanceCriteria,
        showCategories,
        startDate,
        endDate
      });
      setEmails({
        important: results.important_emails,
        nonImportant: results.non_important_emails
      });
    } catch (err) {
      setError(err.message || 'Failed to scan emails');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearCache = async () => {
    try {
      await clearCache();
      setEmails({ important: [], nonImportant: [] });
    } catch (err) {
      setError(err.message || 'Failed to clear cache');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">Gmail Classifier</h1>
              <p className="text-gray-600">Scan and categorize your recent emails</p>
            </div>
            <div className="flex space-x-4">
              <button
                onClick={handleClearCache}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Clear Cache
              </button>
              <button
                onClick={logout}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Sign Out
              </button>
            </div>
          </div>
        </header>

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <ScanControls
            maxResults={maxResults}
            setMaxResults={setMaxResults}
            importanceCriteria={importanceCriteria}
            setImportanceCriteria={setImportanceCriteria}
            showCategories={showCategories}
            setShowCategories={setShowCategories}
            startDate={startDate}
            setStartDate={setStartDate}
            endDate={endDate}
            setEndDate={setEndDate}
            onScan={handleScan}
            isLoading={isLoading}
          />
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <EmailList
          emails={emails}
          showCategories={showCategories}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}

export default Dashboard; 