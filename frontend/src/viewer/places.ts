/** Named places to label over a GPS-placed scene. Coordinates from OpenStreetMap (Overpass,
 * Nov 2026: the feature's node, or its outline's centre); `h` is the height of the dot above
 * the terrain, metres (the top of a tower or statue, a roof; 0 or left out: the ground). */
export type Place = { name: string; note?: string; lat: number; lon: number; h?: number; major?: boolean };

export const PLACES: Place[] = [
  // Landmarks, labelled from far away.
  { name: "Cascade", note: "stairway · Cafesjian Center", lat: 40.19147, lon: 44.51559, h: 15, major: true },
  { name: "Mother Armenia", note: "monument · 51 m", lat: 40.19516, lon: 44.52467, h: 51, major: true },
  { name: "Republic Square", note: "central square", lat: 40.17803, lon: 44.51288, h: 30, major: true },
  { name: "Opera Theatre", note: "opera and ballet", lat: 40.18576, lon: 44.51507, h: 25, major: true },
  { name: "Matenadaran", note: "manuscript museum", lat: 40.19232, lon: 44.52173, h: 20, major: true },
  { name: "Tsitsernakaberd", note: "Genocide Memorial · 44 m stele", lat: 40.18576, lon: 44.49057, h: 44, major: true },
  { name: "Yerevan TV Tower", note: "311 m", lat: 40.17132, lon: 44.53659, h: 311, major: true },
  { name: "St Gregory the Illuminator Cathedral", note: "cathedral · 2001", lat: 40.17205, lon: 44.51687, h: 54, major: true },
  { name: "Erebuni Fortress", note: "Urartian fortress · 782 BC", lat: 40.14036, lon: 44.5379, h: 5, major: true },
  { name: "Hrazdan Stadium", note: "stadium", lat: 40.18039, lon: 44.49465, h: 25, major: true },

  // Districts.
  { name: "Ajapnyak", note: "district", lat: 40.20484, lon: 44.46136, h: 60 },
  { name: "Arabkir", note: "district", lat: 40.20783, lon: 44.50952, h: 60 },
  { name: "Davtashen", note: "district", lat: 40.21933, lon: 44.48988, h: 60 },
  { name: "Malatia-Sebastia", note: "district", lat: 40.17412, lon: 44.44587, h: 60 },
  { name: "Shengavit", note: "district", lat: 40.15121, lon: 44.48323, h: 60 },
  { name: "Kanaker-Zeytun", note: "district", lat: 40.21094, lon: 44.54122, h: 60 },
  { name: "Nor Nork", note: "district", lat: 40.18823, lon: 44.5679, h: 60 },
  { name: "Avan", note: "district", lat: 40.22025, lon: 44.57251, h: 60 },

  // Around Ajapnyak, where the drone footage starts.
  { name: "Egypt Square", note: "square", lat: 40.20909, lon: 44.46243, h: 3 },
  { name: "Gevorg Chaush Square", note: "square", lat: 40.20883, lon: 44.46811, h: 3 },
  { name: "Buenos Aires Park", note: "park", lat: 40.20593, lon: 44.47888, h: 3 },
  { name: "Haybusak University", note: "university", lat: 40.20008, lon: 44.476, h: 20 },
  { name: "Davtashen Bridge", note: "bridge over the Hrazdan", lat: 40.20964, lon: 44.49332, h: 5 },
  { name: "Hrazdan Bridge", note: "bridge", lat: 40.19158, lon: 44.48242, h: 5 },
  { name: "Barekamutyun", note: "metro", lat: 40.19694, lon: 44.49503, h: 3 },
  { name: "Olympavan", note: "sports centre", lat: 40.2184, lon: 44.49791, h: 15 },

  // The centre.
  { name: "Blue Mosque", note: "mosque · 1766", lat: 40.17786, lon: 44.50556, h: 24 },
  { name: "Saint Sarkis Cathedral", note: "cathedral", lat: 40.17706, lon: 44.50216, h: 30 },
  { name: "Katoghike Church", note: "13th-century chapel", lat: 40.18433, lon: 44.519, h: 10 },
  { name: "History Museum", note: "and National Gallery", lat: 40.17851, lon: 44.51386, h: 20 },
  { name: "Charles Aznavour Square", note: "square", lat: 40.18168, lon: 44.5168, h: 3 },
  { name: "Victory Park", note: "park", lat: 40.19634, lon: 44.52307, h: 3 },
  { name: "Lovers' Park", note: "park", lat: 40.19126, lon: 44.50648, h: 3 },
  { name: "English Park", note: "park", lat: 40.17398, lon: 44.50797, h: 3 },
  { name: "Marshal Baghramyan", note: "metro", lat: 40.19149, lon: 44.5079, h: 3 },
  { name: "American University of Armenia", note: "university", lat: 40.19343, lon: 44.50395, h: 20 },
  { name: "Parajanov Museum", note: "museum", lat: 40.1789, lon: 44.50007, h: 10 },
  { name: "Ararat Brandy Factory", note: "brandy museum", lat: 40.17676, lon: 44.49717, h: 15 },
  { name: "Red Bridge", note: "17th-century bridge", lat: 40.17186, lon: 44.50003, h: 3 },
  { name: "Erivan Fortress", note: "ruins", lat: 40.16864, lon: 44.50216, h: 3 },
  { name: "Republican Stadium", note: "stadium", lat: 40.17194, lon: 44.52555, h: 20 },
  { name: "Yerevan Zoo", note: "zoo", lat: 40.19476, lon: 44.55188, h: 3 },
  { name: "Komitas Museum", note: "and the Pantheon", lat: 40.1624, lon: 44.50242, h: 10 },
  { name: "David of Sasun", note: "statue · railway station", lat: 40.15526, lon: 44.50952, h: 15 },

  // Further out.
  { name: "Shengavit Settlement", note: "Bronze Age settlement", lat: 40.15674, lon: 44.47729, h: 3 },
  { name: "Teishebaini", note: "Urartian fortress (Karmir Blur)", lat: 40.15348, lon: 44.45069, h: 5 },
  { name: "Saint Hakob Church", note: "Kanaker", lat: 40.22033, lon: 44.53497, h: 12 },
  { name: "Cathedral of Avan", note: "6th century", lat: 40.21506, lon: 44.57195, h: 8 },
];
