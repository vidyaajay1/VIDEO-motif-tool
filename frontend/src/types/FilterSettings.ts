export type FilterSettings = {
  openChromatin: boolean;
  bindingPeaks: boolean;
  sortByHits: boolean;
  sortByScore: boolean;
  selectedMotif: string;
  perMotifPvals: Record<string, number>;
};