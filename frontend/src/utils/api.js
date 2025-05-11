import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json'
  }
});

export const scanEmails = async ({ maxResults, importanceCriteria, showCategories, startDate, endDate }) => {
  const response = await api.post('/scan_emails', {
    max_results: maxResults,
    importance_prompt: importanceCriteria,
    show_categories: showCategories,
    start_date: startDate,
    end_date: endDate
  });
  return response.data;
};

export const checkAuth = async () => {
  const response = await api.get('/auth/check');
  return response.data;
};

export const login = async (provider) => {
  const response = await api.post('/auth/login', { provider });
  return response.data;
};

export const logout = async () => {
  const response = await api.post('/auth/logout');
  return response.data;
};

export const clearCache = async () => {
  const response = await api.post('/clear_cache');
  return response.data;
}; 