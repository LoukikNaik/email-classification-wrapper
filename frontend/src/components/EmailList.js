import React from 'react';
import EmailCard from './EmailCard';

function EmailList({ emails, showCategories, isLoading }) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center space-x-2">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="text-gray-600">Scanning emails...</span>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {emails.important.length > 0 && (
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Important Emails</h2>
          <div className="space-y-4">
            {emails.important.map((email) => (
              <EmailCard
                key={email.id}
                email={email}
                showCategory={showCategories}
              />
            ))}
          </div>
        </div>
      )}

      {emails.nonImportant.length > 0 && (
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Other Emails</h2>
          <div className="space-y-4">
            {emails.nonImportant.map((email) => (
              <EmailCard
                key={email.id}
                email={email}
                showCategory={showCategories}
              />
            ))}
          </div>
        </div>
      )}

      {emails.important.length === 0 && emails.nonImportant.length === 0 && (
        <div className="text-center text-gray-500">
          No emails found. Click "Scan Emails" to start.
        </div>
      )}
    </div>
  );
}

export default EmailList; 