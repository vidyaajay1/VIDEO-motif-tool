export type FilterSettings = {
  openChromatin: boolean;
  bindingPeaks: boolean;
  sortByHits: boolean;
  sortByScore: boolean;
  selectedMotif: string;
  bestTranscript:boolean;
  perMotifPvals: Record<string, number>;
};