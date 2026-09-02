import "server-only";

import {
  db,
  items,
  libraries,
  servers,
  sessions,
  users,
} from "@streamystats/database";
import type { Item } from "@streamystats/database/schema";
import { tool } from "ai";
import {
  and,
  cosineDistance,
  desc,
  eq,
  ilike,
  inArray,
  isNotNull,
  sql,
} from "drizzle-orm";
import { z } from "zod";
import { getHistoryByFilters } from "@/lib/db/history";
import {
  getSimilarItemsForItem,
  getSimilarStatistics,
} from "@/lib/db/similar-statistics";
import { getMostWatchedItems } from "@/lib/db/statistics";
import {
  getUserStatsSummaryForServer,
  getUsers,
  getUserWatchStats,
} from "@/lib/db/users";
import {
  addItemToWatchlist as addItemToWatchlistDb,
  createWatchlist as createWatchlistDb,
  deleteWatchlist as deleteWatchlistDb,
  getUserOwnWatchlists,
  getWatchlistWithItems,
  removeItemFromWatchlist as removeItemFromWatchlistDb,
  reorderWatchlistItems as reorderWatchlistItemsDb,
  updateWatchlist as updateWatchlistDb,
} from "@/lib/db/watchlists";

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

type FormattableItem = Pick<
  Item,
  | "id"
  | "name"
  | "productionYear"
  | "communityRating"
  | "genres"
  | "primaryImageTag"
  | "seriesId"
  | "seriesPrimaryImageTag"
> & {
  type: string | null;
  overview?: string | null;
};

function formatItem(
  item: FormattableItem,
  stats?: { playCount?: number; playDuration?: number },
) {
  const base = {
    id: item.id,
    name: item.name,
    type: item.type,
    year: item.productionYear,
    rating: item.communityRating,
    genres: item.genres,
    overview: item.overview?.slice(0, 200),
    primaryImageTag: item.primaryImageTag,
    seriesId: item.seriesId,
    seriesPrimaryImageTag: item.seriesPrimaryImageTag,
  };
  if (stats) {
    return {
      ...base,
      playCount: stats.playCount,
      watchTime: stats.playDuration
        ? formatDuration(stats.playDuration)
        : undefined,
    };
  }
  return base;
}

type EmbeddingProvider = "openai-compatible" | "ollama";

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "");
}

async function embedTextForServer({
  serverId,
  text,
}: {
  serverId: number;
  text: string;
}): Promise<
  | { ok: true; embedding: number[] }
  | { ok: false; error: string; reason: "not_configured" | "request_failed" }
> {
  const server = await db.query.servers.findFirst({
    where: eq(servers.id, serverId),
  });

  const provider = server?.embeddingProvider as EmbeddingProvider | null;
  const baseUrl = server?.embeddingBaseUrl ?? null;
  const model = server?.embeddingModel ?? null;
  const apiKey = server?.embeddingApiKey ?? null;
  const dimensions = server?.embeddingDimensions ?? null;

  if (!provider || !baseUrl || !model) {
    return {
      ok: false,
      reason: "not_configured",
      error:
        "Embeddings are not configured for this server. Configure them in Settings > Embeddings.",
    };
  }

  try {
    if (provider === "ollama") {
      const res = await fetch(`${normalizeBaseUrl(baseUrl)}/api/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, input: text }),
      });

      if (!res.ok) {
        return {
          ok: false,
          reason: "request_failed",
          error: `Embedding request failed (status ${res.status})`,
        };
      }

      const json = (await res.json()) as {
        embeddings?: number[][];
      };
      const embedding = json.embeddings?.[0];
      if (!Array.isArray(embedding) || embedding.length === 0) {
        return {
          ok: false,
          reason: "request_failed",
          error: "Embedding request returned no embedding vector",
        };
      }
      return { ok: true, embedding };
    }

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`;
    }

    const body: Record<string, unknown> = { model, input: text };
    if (typeof dimensions === "number" && dimensions > 0) {
      body.dimensions = dimensions;
    }

    const normalized = normalizeBaseUrl(baseUrl);
    const embeddingsUrl = normalized.endsWith("/v1")
      ? `${normalized}/embeddings`
      : `${normalized}/v1/embeddings`;
    const res = await fetch(embeddingsUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      return {
        ok: false,
        reason: "request_failed",
        error: `Embedding request failed (status ${res.status})`,
      };
    }

    const json = (await res.json()) as {
      data?: Array<{ embedding?: number[] }>;
    };
    const embedding = json.data?.[0]?.embedding;
    if (!Array.isArray(embedding) || embedding.length === 0) {
      return {
        ok: false,
        reason: "request_failed",
        error: "Embedding request returned no embedding vector",
      };
    }
    return { ok: true, embedding };
  } catch (error) {
    return {
      ok: false,
      reason: "request_failed",
      error:
        error instanceof Error ? error.message : "Embedding request failed",
    };
  }
}

function getHolidayHintScore(item: Item): number {
  const haystack = `${item.name ?? ""}\n${item.overview ?? ""}`.toLowerCase();
  const keywords = [
    "christmas",
    "xmas",
    "holiday",
    "santa",
    "reindeer",
    "elf",
    "grinch",
    "nativity",
    "yuletide",
    "snowman",
    "mistletoe",
  ];
  const keywordHits = keywords.reduce(
    (sum, k) => sum + (haystack.includes(k) ? 1 : 0),
    0,
  );

  const genreHits = Array.isArray(item?.genres)
    ? item.genres.reduce(
        (sum: number, g: string) =>
          sum +
          (typeof g === "string" &&
          ["holiday", "christmas"].includes(g.toLowerCase())
            ? 1
            : 0),
        0,
      )
    : 0;

  return keywordHits + genreHits * 2;
}

const limitSchema = z.object({
  limit: z
    .number()
    .optional()
    .default(10)
    .describe("Number of items to return"),
});

const limitTypeSchema = z.object({
  limit: z
    .number()
    .optional()
    .default(10)
    .describe("Number of items to return"),
  type: z
    .enum(["Movie", "Series", "all"])
    .optional()
    .default("all")
    .describe("Filter by item type"),
});

const sortOrderSchema = z.enum(["custom", "name", "dateAdded", "releaseDate"]);
const allowedItemTypeSchema = z.enum(["Movie", "Series", "Episode"]);

export function createChatTools(serverId: number, userId: string) {
  return {
    getUserMostWatchedMovies: tool({
      description:
        "Get the user's most watched movies ordered by total watch time",
      inputSchema: limitSchema,
      execute: async ({ limit }: z.infer<typeof limitSchema>) => {
        const result = await getMostWatchedItems({ serverId, userId });
        const movies = result.Movie.slice(0, limit);
        return {
          movies: movies.map((m) =>
            formatItem(m, {
              playCount: m.totalPlayCount,
              playDuration: m.totalPlayDuration,
            }),
          ),
          message:
            movies.length > 0
              ? `Found ${movies.length} most watched movies`
              : "No movies watched yet",
        };
      },
    }),

    getUserMostWatchedSeries: tool({
      description:
        "Get the user's most watched TV series ordered by total watch time",
      inputSchema: limitSchema,
      execute: async ({ limit }: z.infer<typeof limitSchema>) => {
        const result = await getMostWatchedItems({ serverId, userId });
        const series = result.Series.slice(0, limit);
        return {
          series: series.map((s) =>
            formatItem(s, {
              playCount: s.totalPlayCount,
              playDuration: s.totalPlayDuration,
            }),
          ),
          message:
            series.length > 0
              ? `Found ${series.length} most watched series`
              : "No series watched yet",
        };
      },
    }),

    getPersonalizedRecommendations: tool({
      description:
        "Get personalized movie and series recommendations based on user's watch history using AI embeddings. Each recommendation includes a 'reason' field (e.g. 'Because you watched X and Y') and a 'basedOn' array with the watched items that led to this recommendation. Always use this data when presenting recommendations to explain what they're based on.",
      inputSchema: limitTypeSchema,
      execute: async ({ limit, type }: z.infer<typeof limitTypeSchema>) => {
        const recommendations = await getSimilarStatistics({
          serverId,
          userId,
          limit: limit * 2,
        });

        const filtered =
          type === "all"
            ? recommendations
            : recommendations.filter((r) => r.item.type === type);

        const enrichedRecs = filtered.slice(0, limit).map((r) => {
          const recGenres = new Set(r.item.genres || []);
          const basedOnItems = r.basedOn.slice(0, 3);

          const sharedGenres = basedOnItems.flatMap((b) =>
            (b.genres || []).filter((g) => recGenres.has(g)),
          );
          const uniqueSharedGenres = [...new Set(sharedGenres)];

          let reason = "";
          if (basedOnItems.length > 0) {
            const baseNames = basedOnItems.map((b) => b.name);
            if (basedOnItems.length === 1) {
              reason = `Because you watched "${baseNames[0]}"`;
            } else {
              reason = `Because you watched "${baseNames
                .slice(0, -1)
                .join('", "')}" and "${baseNames[baseNames.length - 1]}"`;
            }
            if (uniqueSharedGenres.length > 0) {
              reason += ` (shared: ${uniqueSharedGenres
                .slice(0, 3)
                .join(", ")})`;
            }
          } else {
            reason = "Popular on this server";
          }

          return {
            ...formatItem(r.item),
            similarityPercent: Math.round(r.similarity * 100),
            reason,
            basedOn: basedOnItems.map((b) => ({
              name: b.name,
              type: b.type,
              genres: b.genres?.slice(0, 3),
            })),
            sharedGenres: uniqueSharedGenres.slice(0, 5),
          };
        });

        return {
          recommendations: enrichedRecs,
          message:
            enrichedRecs.length > 0
              ? `Found ${enrichedRecs.length} personalized recommendations with reasoning`
              : "Unable to generate recommendations. Make sure embeddings are configured and you have watch history.",
        };
      },
    }),

    getRecentlyAddedItems: tool({
      description: "Get recently added movies and series to the library",
      inputSchema: z.object({
        limit: z
          .number()
          .optional()
          .default(20)
          .describe("Number of items to return"),
        type: z
          .enum(["Movie", "Series", "all"])
          .optional()
          .default("all")
          .describe("Filter by item type"),
      }),
      execute: async ({
        limit,
        type,
      }: {
        limit: number;
        type: "Movie" | "Series" | "all";
      }) => {
        const conditions = [
          eq(items.serverId, serverId),
          isNotNull(items.dateCreated),
        ];
        if (type !== "all") {
          conditions.push(eq(items.type, type));
        } else {
          conditions.push(inArray(items.type, ["Movie", "Series"]));
        }

        const recentItems = await db
          .select()
          .from(items)
          .where(and(...conditions))
          .orderBy(desc(items.dateCreated))
          .limit(limit);

        return {
          items: recentItems.map((item) => ({
            ...formatItem(item),
            addedDate: item.dateCreated?.toISOString().split("T")[0],
          })),
          message: `Found ${recentItems.length} recently added ${
            type === "all" ? "items" : `${type.toLowerCase()}s`
          }`,
        };
      },
    }),

    searchItems: tool({
      description: "Search for movies and series by name or genre",
      inputSchema: z.object({
        query: z.string().describe("Search query for item name"),
        type: z
          .enum(["Movie", "Series", "all"])
          .optional()
          .default("all")
          .describe("Filter by type"),
        limit: z.number().optional().default(20).describe("Number of results"),
      }),
      execute: async ({
        query,
        type,
        limit,
      }: {
        query: string;
        type: "Movie" | "Series" | "all";
        limit: number;
      }) => {
        const conditions = [
          eq(items.serverId, serverId),
          ilike(items.name, `%${query}%`),
        ];
        if (type !== "all") {
          conditions.push(eq(items.type, type));
        } else {
          conditions.push(inArray(items.type, ["Movie", "Series"]));
        }

        const results = await db
          .select()
          .from(items)
          .where(and(...conditions))
          .orderBy(desc(items.communityRating))
          .limit(limit);

        return {
          items: results.map((item) => formatItem(item)),
          message:
            results.length > 0
              ? `Found ${results.length} items matching "${query}"`
              : `No items found matching "${query}"`,
        };
      },
    }),

    getUserWatchStatistics: tool({
      description:
        "Get overall watch statistics for the user including total watch time and streaks",
      inputSchema: z.object({}),
      execute: async () => {
        const stats = await getUserWatchStats({ serverId, userId });
        return {
          totalWatchTime: formatDuration(stats.total_watch_time),
          totalWatchTimeSeconds: stats.total_watch_time,
          totalPlays: stats.total_plays,
          longestStreak: stats.longest_streak,
          message: `User has watched ${formatDuration(
            stats.total_watch_time,
          )} total with ${stats.total_plays} plays`,
        };
      },
    }),

    getWatchtimeByInterval: tool({
      description:
        "Get watchtime statistics for users within a specific date range. Can filter by user and item type. Returns results sorted by watchtime (highest first). ALWAYS use this tool when asked about watchtime for a specific time period (yesterday, last week, this month, etc.) or when asked 'who watched the most' for any time period. Examples: 'who watched the most yesterday?', 'how much did user X watch last week?', 'who watched the most movies this month?'",
      inputSchema: z.object({
        startDate: z.string().describe("Start date in ISO format (YYYY-MM-DD)"),
        endDate: z.string().describe("End date in ISO format (YYYY-MM-DD)"),
        userId: z
          .string()
          .optional()
          .describe(
            "Optional user ID to filter by specific user. If not provided, returns data for all users.",
          ),
        itemType: z
          .enum(["Movie", "Series", "Episode", "all"])
          .optional()
          .default("all")
          .describe("Filter by item type. Defaults to 'all'."),
      }),
      execute: async ({ startDate, endDate, userId, itemType }) => {
        const results = await getUserStatsSummaryForServer({
          serverId,
          startDate,
          endDate,
          userId,
          itemType: itemType || "all",
        });

        return {
          users: results.map((user) => ({
            userId: user.userId,
            userName: user.userName,
            watchTime: formatDuration(user.totalWatchTime),
            watchTimeSeconds: user.totalWatchTime,
            playCount: user.sessionCount,
          })),
          message:
            results.length > 0
              ? `Found ${results.length} user${
                  results.length === 1 ? "" : "s"
                } with watchtime data`
              : "No watchtime data found for the specified criteria",
        };
      },
    }),

    getHistoryByFilters: tool({
      description:
        "Get playback history with filters for user, item type, and time interval. Use this to see what specific items were watched during a time period. Can be combined with getUserWatchStatistics to get detailed viewing information.",
      inputSchema: z.object({
        userId: z
          .string()
          .optional()
          .describe(
            "Optional user ID to filter by specific user. If not provided, returns history for all users.",
          ),
        itemType: z
          .enum(["Movie", "Series", "Episode", "all"])
          .optional()
          .default("all")
          .describe("Filter by item type. Defaults to 'all'."),
        startDate: z
          .string()
          .optional()
          .describe("Start date in ISO format (YYYY-MM-DD). Optional."),
        endDate: z
          .string()
          .optional()
          .describe("End date in ISO format (YYYY-MM-DD). Optional."),
        limit: z
          .number()
          .optional()
          .default(50)
          .describe(
            "Maximum number of history items to return. Defaults to 50.",
          ),
      }),
      execute: async ({ userId, itemType, startDate, endDate, limit }) => {
        const history = await getHistoryByFilters({
          serverId,
          userId,
          itemType: itemType || "all",
          startDate,
          endDate,
          limit: limit || 50,
        });

        return {
          history: history.map((item) => {
            const seriesId =
              item.item?.seriesId || item.session.seriesId || null;
            const seriesName =
              item.item?.seriesName || item.session.seriesName || null;
            const seasonId =
              item.item?.seasonId || item.session.seasonId || null;
            const seasonName = item.item?.seasonName || null;
            const episodeNumber = item.item?.indexNumber || null;
            const seasonNumber = item.item?.parentIndexNumber || null;

            return {
              itemName: item.item?.name || item.session.itemName || "Unknown",
              itemId: item.item?.id || item.session.itemId,
              itemType: item.item?.type || "Unknown",
              userName: item.user?.name || item.session.userName || "Unknown",
              userId: item.user?.id || item.session.userId,
              watchDate: item.session.startTime
                ? item.session.startTime.toISOString()
                : null,
              watchDuration: item.session.playDuration || 0,
              watchDurationFormatted: formatDuration(
                item.session.playDuration || 0,
              ),
              completionPercentage: item.session.percentComplete || 0,
              deviceName: item.session.deviceName,
              clientName: item.session.clientName,
              seriesId,
              seriesName,
              seasonId,
              seasonName,
              episodeNumber,
              seasonNumber,
              ...(seriesName && {
                displayName: `${seriesName}${
                  seasonNumber && episodeNumber
                    ? ` - S${seasonNumber}E${episodeNumber}`
                    : seasonNumber
                      ? ` - Season ${seasonNumber}`
                      : ""
                } - ${item.item?.name || item.session.itemName || "Unknown"}`,
              }),
            };
          }),
          message:
            history.length > 0
              ? `Found ${history.length} history item${
                  history.length === 1 ? "" : "s"
                }`
              : "No history found for the specified criteria",
        };
      },
    }),

    getAvailableUsers: tool({
      description:
        "Get list of all users on this server (for finding users to get shared recommendations with)",
      inputSchema: z.object({}),
      execute: async () => {
        const allUsers = await getUsers({ serverId });
        return {
          users: allUsers
            .filter((u) => !u.isHidden && !u.isDisabled)
            .map((u) => ({ id: u.id, name: u.name })),
          message: `Found ${allUsers.length} users`,
        };
      },
    }),

    getSharedRecommendations: tool({
      description:
        "Get movie/series recommendations that both the current user and another user would enjoy based on their overlapping watch history",
      inputSchema: z.object({
        otherUserName: z
          .string()
          .describe(
            "Name of the other user to find shared recommendations with",
          ),
        limit: z
          .number()
          .optional()
          .default(10)
          .describe("Number of recommendations"),
      }),
      execute: async ({
        otherUserName,
        limit,
      }: {
        otherUserName: string;
        limit: number;
      }) => {
        const otherUser = await db.query.users.findFirst({
          where: and(
            eq(users.serverId, serverId),
            ilike(users.name, `%${otherUserName}%`),
          ),
        });

        if (!otherUser) {
          return {
            recommendations: [],
            message: `Could not find user "${otherUserName}"`,
          };
        }

        const [currentUserRecs, otherUserRecs] = await Promise.all([
          getSimilarStatistics({ serverId, userId, limit: 50 }),
          getSimilarStatistics({
            serverId,
            userId: otherUser.id,
            limit: 50,
          }),
        ]);

        const currentUserRecIds = new Set(
          currentUserRecs.map((r) => r.item.id),
        );
        const sharedRecs = otherUserRecs
          .filter((r) => currentUserRecIds.has(r.item.id))
          .slice(0, limit);

        if (sharedRecs.length < limit) {
          const currentUserWatched = await db
            .select({ itemId: sessions.itemId })
            .from(sessions)
            .where(
              and(eq(sessions.serverId, serverId), eq(sessions.userId, userId)),
            )
            .groupBy(sessions.itemId);

          const otherUserWatched = await db
            .select({ itemId: sessions.itemId })
            .from(sessions)
            .where(
              and(
                eq(sessions.serverId, serverId),
                eq(sessions.userId, otherUser.id),
              ),
            )
            .groupBy(sessions.itemId);

          const currentWatchedIds = new Set(
            currentUserWatched.map((w) => w.itemId),
          );
          const bothWatched = otherUserWatched
            .filter((w) => w.itemId && currentWatchedIds.has(w.itemId))
            .map((w) => w.itemId)
            .filter(Boolean) as string[];

          if (bothWatched.length > 0) {
            const sharedGenres = await db
              .select({ genres: items.genres })
              .from(items)
              .where(inArray(items.id, bothWatched.slice(0, 20)));

            const genreCounts = new Map<string, number>();
            for (const item of sharedGenres) {
              if (item.genres) {
                for (const genre of item.genres) {
                  genreCounts.set(genre, (genreCounts.get(genre) || 0) + 1);
                }
              }
            }

            const topGenres = [...genreCounts.entries()]
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([genre]) => genre);

            if (topGenres.length > 0) {
              const existingIds = new Set([
                ...sharedRecs.map((r) => r.item.id),
                ...bothWatched,
              ]);

              const genreRecs = await db
                .select()
                .from(items)
                .where(
                  and(
                    eq(items.serverId, serverId),
                    inArray(items.type, ["Movie", "Series"]),
                    sql`${items.genres} && ARRAY[${sql.join(
                      topGenres.map((g) => sql`${g}`),
                      sql`, `,
                    )}]::text[]`,
                  ),
                )
                .orderBy(desc(items.communityRating))
                .limit(limit - sharedRecs.length + 10);

              const additionalRecs = genreRecs
                .filter((item) => !existingIds.has(item.id))
                .slice(0, limit - sharedRecs.length);

              return {
                recommendations: [
                  ...sharedRecs.map((r) => ({
                    ...formatItem(r.item),
                    sharedMatch: true,
                  })),
                  ...additionalRecs.map((item) => ({
                    ...formatItem(item),
                    sharedMatch: false,
                    basedOnSharedGenres: topGenres,
                  })),
                ],
                otherUser: otherUser.name,
                sharedGenres: topGenres,
                message: `Found recommendations for you and ${otherUser.name}`,
              };
            }
          }
        }

        return {
          recommendations: sharedRecs.map((r) => ({
            ...formatItem(r.item),
            similarity: Math.round(r.similarity * 100),
          })),
          otherUser: otherUser.name,
          message:
            sharedRecs.length > 0
              ? `Found ${sharedRecs.length} shared recommendations for you and ${otherUser.name}`
              : "No strong shared recommendations found. Try watching more content together!",
        };
      },
    }),

    getLibraries: tool({
      description: "Get list of media libraries on the server",
      inputSchema: z.object({}),
      execute: async () => {
        const libs = await db
          .select()
          .from(libraries)
          .where(eq(libraries.serverId, serverId));

        return {
          libraries: libs.map((l) => ({
            id: l.id,
            name: l.name,
            type: l.type,
          })),
          message: `Found ${libs.length} libraries`,
        };
      },
    }),

    getSimilarToItem: tool({
      description:
        "Get items similar to a specific movie or series. Returns a 'sourceItem' showing what the search was based on, and similar items with a 'reason' field explaining the connection. Use when user asks 'what should I watch after X' or 'find movies like X'. Always mention the sourceItem when presenting results.",
      inputSchema: z.object({
        itemName: z
          .string()
          .describe("Name of the movie or series to find similar items for"),
        limit: z
          .number()
          .optional()
          .default(10)
          .describe("Number of similar items to return"),
      }),
      execute: async ({
        itemName,
        limit,
      }: {
        itemName: string;
        limit: number;
      }) => {
        const foundItems = await db
          .select()
          .from(items)
          .where(
            and(
              eq(items.serverId, serverId),
              ilike(items.name, `%${itemName}%`),
              inArray(items.type, ["Movie", "Series"]),
            ),
          )
          .orderBy(desc(items.communityRating))
          .limit(1);

        if (foundItems.length === 0) {
          return {
            similar: [],
            message: `Could not find "${itemName}" in the library`,
          };
        }

        const sourceItem = foundItems[0];
        const similarItems = await getSimilarItemsForItem(
          serverId,
          sourceItem.id,
          limit,
        );

        const enrichedSimilar = similarItems.map((r) => {
          const sourceGenres = new Set(sourceItem.genres || []);
          const recGenres = r.item.genres || [];
          const sharedGenres = recGenres.filter((g) => sourceGenres.has(g));

          let reason = `Similar to "${sourceItem.name}"`;
          if (sharedGenres.length > 0) {
            reason += ` - both are ${sharedGenres.slice(0, 3).join(", ")} ${
              r.item.type === "Movie" ? "movies" : "series"
            }`;
          }
          if (sourceItem.productionYear && r.item.productionYear) {
            const yearDiff = Math.abs(
              sourceItem.productionYear - r.item.productionYear,
            );
            if (yearDiff <= 5) {
              reason += `, from the same era (${r.item.productionYear})`;
            }
          }

          return {
            ...formatItem(r.item),
            similarityPercent: Math.round(r.similarity * 100),
            reason,
            sharedGenres,
          };
        });

        return {
          sourceItem: formatItem(sourceItem),
          similar: enrichedSimilar,
          message:
            enrichedSimilar.length > 0
              ? `Found ${enrichedSimilar.length} items similar to "${sourceItem.name}"`
              : `No similar items found for "${sourceItem.name}". Embeddings may not be configured.`,
        };
      },
    }),

    getItemsByGenre: tool({
      description: "Get movies or series filtered by genre",
      inputSchema: z.object({
        genre: z
          .string()
          .describe("Genre to filter by (e.g., 'Action', 'Comedy', 'Drama')"),
        type: z.enum(["Movie", "Series", "all"]).optional().default("all"),
        limit: z.number().optional().default(20),
      }),
      execute: async ({
        genre,
        type,
        limit,
      }: {
        genre: string;
        type: "Movie" | "Series" | "all";
        limit: number;
      }) => {
        const conditions = [
          eq(items.serverId, serverId),
          sql`${genre} = ANY(${items.genres})`,
        ];
        if (type !== "all") {
          conditions.push(eq(items.type, type));
        } else {
          conditions.push(inArray(items.type, ["Movie", "Series"]));
        }

        const results = await db
          .select()
          .from(items)
          .where(and(...conditions))
          .orderBy(desc(items.communityRating))
          .limit(limit);

        return {
          items: results.map((item) => formatItem(item)),
          genre,
          message:
            results.length > 0
              ? `Found ${results.length} ${
                  type === "all" ? "items" : `${type.toLowerCase()}s`
                } in ${genre}`
              : `No items found in genre "${genre}"`,
        };
      },
    }),

    getTopRatedItems: tool({
      description: "Get top rated movies or series by community rating",
      inputSchema: z.object({
        type: z.enum(["Movie", "Series", "all"]).optional().default("all"),
        limit: z.number().optional().default(20),
        minRating: z
          .number()
          .optional()
          .default(7)
          .describe("Minimum rating (0-10)"),
      }),
      execute: async ({
        type,
        limit,
        minRating,
      }: {
        type: "Movie" | "Series" | "all";
        limit: number;
        minRating: number;
      }) => {
        const conditions = [
          eq(items.serverId, serverId),
          isNotNull(items.communityRating),
          sql`${items.communityRating} >= ${minRating}`,
        ];
        if (type !== "all") {
          conditions.push(eq(items.type, type));
        } else {
          conditions.push(inArray(items.type, ["Movie", "Series"]));
        }

        const results = await db
          .select()
          .from(items)
          .where(and(...conditions))
          .orderBy(desc(items.communityRating))
          .limit(limit);

        return {
          items: results.map((item) => formatItem(item)),
          message: `Found ${results.length} top-rated items (${minRating}+ rating)`,
        };
      },
    }),

    searchLibraryBySemanticQuery: tool({
      description:
        "Semantic search across the user's library using embeddings. Use this for theme-based queries like 'Christmas movie I have', 'cozy winter movie', or 'something like a heist thriller'. Returns items from the library ranked by semantic similarity.",
      inputSchema: z.object({
        query: z
          .string()
          .describe("What the user is looking for (theme/query)"),
        type: z
          .enum(["Movie", "Series", "all"])
          .optional()
          .default("Movie")
          .describe("Filter by item type"),
        limit: z.number().optional().default(10).describe("Number of results"),
      }),
      execute: async ({
        query,
        type,
        limit,
      }: {
        query: string;
        type: "Movie" | "Series" | "all";
        limit: number;
      }) => {
        const embed = await embedTextForServer({
          serverId,
          text: query,
        });

        // Fallback: embeddings not configured, do a best-effort keyword/genre search.
        if (!embed.ok) {
          const lowered = query.toLowerCase();
          const isHolidayQuery =
            lowered.includes("christmas") ||
            lowered.includes("xmas") ||
            lowered.includes("holiday");

          const conditions = [eq(items.serverId, serverId)];
          if (type !== "all") {
            conditions.push(eq(items.type, type));
          } else {
            conditions.push(inArray(items.type, ["Movie", "Series"]));
          }

          if (isHolidayQuery) {
            conditions.push(
              sql`(${items.name} ILIKE '%christmas%' OR ${items.overview} ILIKE '%christmas%' OR ${items.name} ILIKE '%holiday%' OR ${items.overview} ILIKE '%holiday%' OR ${items.name} ILIKE '%xmas%' OR ${items.overview} ILIKE '%xmas%' OR ${items.genres} && ARRAY['Holiday','Christmas']::text[])`,
            );
          } else {
            conditions.push(
              sql`(${items.name} ILIKE ${`%${query}%`} OR ${
                items.overview
              } ILIKE ${`%${query}%`})`,
            );
          }

          const results = await db
            .select()
            .from(items)
            .where(and(...conditions))
            .orderBy(desc(items.communityRating))
            .limit(limit);

          return {
            items: results.map((item) => ({
              ...formatItem(item),
              reason: `Matched text/metadata for "${query}"`,
            })),
            message:
              results.length > 0
                ? `Found ${results.length} items matching "${query}" (fallback search)`
                : `No items found matching "${query}". ${
                    embed.reason === "not_configured" ? embed.error : ""
                  }`.trim(),
            usedEmbeddings: false,
          };
        }

        const similarity = sql<number>`1 - (${cosineDistance(
          items.embedding,
          embed.embedding,
        )})`;

        const conditions = [
          eq(items.serverId, serverId),
          isNotNull(items.embedding),
        ];
        if (type !== "all") {
          conditions.push(eq(items.type, type));
        } else {
          conditions.push(inArray(items.type, ["Movie", "Series"]));
        }

        const candidates = await db
          .select({
            item: items,
            similarity,
          })
          .from(items)
          .where(and(...conditions))
          .orderBy(desc(similarity), desc(items.communityRating))
          .limit(Math.max(limit * 12, 50));

        const ranked = candidates
          .map((c) => {
            const sim = Number(c.similarity);
            const hint = getHolidayHintScore(c.item);
            const bonus = Math.min(0.12, hint * 0.02);
            return {
              ...c,
              similarity: sim,
              holidayHintScore: hint,
              finalScore: sim + bonus,
            };
          })
          .sort((a, b) => b.finalScore - a.finalScore)
          .slice(0, limit);

        return {
          items: ranked.map((r) => ({
            ...formatItem(r.item),
            similarityPercent: Math.round(r.similarity * 100),
            reason: `Semantic match for "${query}"`,
          })),
          message:
            ranked.length > 0
              ? `Found ${ranked.length} semantically similar items for "${query}"`
              : `No items found for "${query}" (embeddings search)`,
          usedEmbeddings: true,
        };
      },
    }),

    getMyWatchlists: tool({
      description:
        "List the current user's own watchlists (the only ones they can edit). Use this to resolve a watchlist name mentioned by the user into its numeric ID before creating, updating, deleting, or adding/removing items.",
      inputSchema: z.object({}),
      execute: async () => {
        const lists = await getUserOwnWatchlists({ serverId, userId });
        return {
          watchlists: lists.map((w) => ({
            id: w.id,
            name: w.name,
            description: w.description,
            isPublic: w.isPublic,
            allowedItemType: w.allowedItemType,
            defaultSortOrder: w.defaultSortOrder,
            itemCount: w.itemCount,
          })),
          message:
            lists.length > 0
              ? `Found ${lists.length} watchlist(s)`
              : "You have no watchlists yet",
        };
      },
    }),

    getWatchlistContents: tool({
      description:
        "Get the metadata and items of one of the user's watchlists. Returns item IDs, which are needed to remove an item. Also use this to answer questions like 'what's in my Sci-Fi list?'.",
      inputSchema: z.object({
        watchlistId: z
          .number()
          .describe("The numeric watchlist ID (from getMyWatchlists)"),
      }),
      execute: async ({ watchlistId }: { watchlistId: number }) => {
        const list = await getWatchlistWithItems({ watchlistId, userId });
        if (!list) {
          return {
            success: false,
            message: "Watchlist not found or not accessible",
          };
        }
        return {
          success: true,
          watchlist: {
            id: list.id,
            name: list.name,
            description: list.description,
            isPublic: list.isPublic,
            allowedItemType: list.allowedItemType,
            defaultSortOrder: list.defaultSortOrder,
          },
          items: list.items.map((wi) => ({
            ...formatItem(wi.item),
            watchlistItemPosition: wi.position,
          })),
          message: `"${list.name}" has ${list.items.length} item(s)`,
        };
      },
    }),

    createWatchlist: tool({
      description:
        "Create a new watchlist owned by the current user. Does not need confirmation. New watchlists cannot be promoted/featured from chat.",
      inputSchema: z.object({
        name: z.string().trim().min(1).describe("Watchlist name"),
        description: z.string().optional().describe("Optional description"),
        isPublic: z
          .boolean()
          .optional()
          .default(false)
          .describe("Whether other users on the server can see this list"),
        allowedItemType: allowedItemTypeSchema
          .optional()
          .describe("If set, only items of this type can be added"),
        defaultSortOrder: sortOrderSchema
          .optional()
          .default("custom")
          .describe("Default ordering for items in the list"),
      }),
      execute: async ({
        name,
        description,
        isPublic,
        allowedItemType,
        defaultSortOrder,
      }: {
        name: string;
        description?: string;
        isPublic: boolean;
        allowedItemType?: "Movie" | "Series" | "Episode";
        defaultSortOrder: "custom" | "name" | "dateAdded" | "releaseDate";
      }) => {
        const created = await createWatchlistDb({
          serverId,
          userId,
          name: name.trim(),
          description: description ?? null,
          isPublic: isPublic ?? false,
          allowedItemType: allowedItemType ?? null,
          defaultSortOrder: defaultSortOrder ?? "custom",
        });
        return {
          success: true,
          watchlist: {
            id: created.id,
            name: created.name,
            description: created.description,
            isPublic: created.isPublic,
            allowedItemType: created.allowedItemType,
            defaultSortOrder: created.defaultSortOrder,
          },
          message: `Created watchlist "${created.name}"`,
        };
      },
    }),

    updateWatchlist: tool({
      description:
        "Update metadata of one of the user's own watchlists (name, description, visibility, item-type lock, sort order). Only the fields you provide are changed. Does not need confirmation.",
      inputSchema: z.object({
        watchlistId: z
          .number()
          .describe("The numeric watchlist ID (from getMyWatchlists)"),
        name: z.string().trim().min(1).optional().describe("New name"),
        description: z
          .string()
          .nullable()
          .optional()
          .describe("New description, or null to clear it"),
        isPublic: z.boolean().optional().describe("New visibility"),
        allowedItemType: allowedItemTypeSchema
          .nullable()
          .optional()
          .describe("New item-type lock, or null to remove the lock"),
        defaultSortOrder: sortOrderSchema
          .optional()
          .describe("New default sort order"),
      }),
      execute: async ({
        watchlistId,
        name,
        description,
        isPublic,
        allowedItemType,
        defaultSortOrder,
      }: {
        watchlistId: number;
        name?: string;
        description?: string | null;
        isPublic?: boolean;
        allowedItemType?: "Movie" | "Series" | "Episode" | null;
        defaultSortOrder?: "custom" | "name" | "dateAdded" | "releaseDate";
      }) => {
        const data: Parameters<typeof updateWatchlistDb>[0]["data"] = {};
        if (name !== undefined) data.name = name.trim();
        if (description !== undefined) data.description = description;
        if (isPublic !== undefined) data.isPublic = isPublic;
        if (allowedItemType !== undefined)
          data.allowedItemType = allowedItemType;
        if (defaultSortOrder !== undefined)
          data.defaultSortOrder = defaultSortOrder;

        if (Object.keys(data).length === 0) {
          return { success: false, message: "No fields to update" };
        }

        const updated = await updateWatchlistDb({ watchlistId, userId, data });
        if (!updated) {
          return {
            success: false,
            message: "Watchlist not found or you do not own it",
          };
        }
        return {
          success: true,
          watchlist: {
            id: updated.id,
            name: updated.name,
            description: updated.description,
            isPublic: updated.isPublic,
            allowedItemType: updated.allowedItemType,
            defaultSortOrder: updated.defaultSortOrder,
          },
          message: `Updated watchlist "${updated.name}"`,
        };
      },
    }),

    deleteWatchlist: tool({
      description:
        "Permanently delete one of the user's own watchlists and all of its items. This cannot be undone — state exactly which watchlist will be deleted and wait for the user to confirm before calling this.",
      inputSchema: z.object({
        watchlistId: z
          .number()
          .describe("The numeric watchlist ID (from getMyWatchlists)"),
      }),
      execute: async ({ watchlistId }: { watchlistId: number }) => {
        const deleted = await deleteWatchlistDb({ watchlistId, userId });
        return {
          success: deleted,
          message: deleted
            ? "Watchlist deleted"
            : "Watchlist not found or you do not own it",
        };
      },
    }),

    addItemToWatchlist: tool({
      description:
        "Add a library item to one of the user's own watchlists. Get a real item ID from a search or recommendation tool first — never invent IDs. Does not need confirmation.",
      inputSchema: z.object({
        watchlistId: z
          .number()
          .describe("The numeric watchlist ID (from getMyWatchlists)"),
        itemId: z
          .string()
          .min(1)
          .describe("The library item ID (from a search/recommendation tool)"),
      }),
      execute: async ({
        watchlistId,
        itemId,
      }: {
        watchlistId: number;
        itemId: string;
      }) => {
        const trimmedItemId = itemId.trim();

        const ownedLists = await getUserOwnWatchlists({ serverId, userId });
        if (!ownedLists.some((l) => l.id === watchlistId)) {
          return {
            success: false,
            message: "Watchlist not found or you do not own it",
          };
        }

        const [libraryItem] = await db
          .select({ id: items.id })
          .from(items)
          .where(and(eq(items.id, trimmedItemId), eq(items.serverId, serverId)))
          .limit(1);
        if (!libraryItem) {
          return {
            success: false,
            message: "That item isn't in this server's library",
          };
        }

        const result = await addItemToWatchlistDb({
          watchlistId,
          itemId: trimmedItemId,
          userId,
        });
        if (!result) {
          return {
            success: false,
            message:
              "Could not add item — the watchlist may not exist, you may not own it, the item type may not be allowed by this list, the item isn't in the library, or it's already in the list.",
          };
        }
        return { success: true, message: "Added item to watchlist" };
      },
    }),

    removeItemFromWatchlist: tool({
      description:
        "Remove an item from one of the user's own watchlists. State which item and watchlist and wait for the user to confirm before calling this.",
      inputSchema: z.object({
        watchlistId: z
          .number()
          .describe("The numeric watchlist ID (from getMyWatchlists)"),
        itemId: z
          .string()
          .min(1)
          .describe(
            "The library item ID to remove (from getWatchlistContents)",
          ),
      }),
      execute: async ({
        watchlistId,
        itemId,
      }: {
        watchlistId: number;
        itemId: string;
      }) => {
        const removed = await removeItemFromWatchlistDb({
          watchlistId,
          itemId: itemId.trim(),
          userId,
        });
        return {
          success: removed,
          message: removed
            ? "Removed item from watchlist"
            : "Item not found in that watchlist or you do not own it",
        };
      },
    }),

    reorderWatchlistItems: tool({
      description:
        "Set the order of items in one of the user's own watchlists. Call getWatchlistContents first, then pass every current item ID exactly once in the desired order (position 0 first). Missing, extra, or duplicate IDs are rejected. Only affects lists sorted by 'custom' order. Does not need confirmation.",
      inputSchema: z.object({
        watchlistId: z
          .number()
          .describe("The numeric watchlist ID (from getMyWatchlists)"),
        itemIds: z
          .array(z.string().min(1))
          .min(1)
          .describe(
            "Every current item ID exactly once, in the desired order (first position first)",
          ),
      }),
      execute: async ({
        watchlistId,
        itemIds,
      }: {
        watchlistId: number;
        itemIds: string[];
      }) => {
        const trimmed = itemIds.map((id) => id.trim());
        const provided = new Set(trimmed);
        if (provided.size !== trimmed.length) {
          return {
            success: false,
            message: "The item ID list contains duplicates",
          };
        }

        const list = await getWatchlistWithItems({ watchlistId, userId });
        if (!list) {
          return {
            success: false,
            message: "Watchlist not found or you do not own it",
          };
        }

        const current = new Set(list.items.map((i) => i.item.id));
        if (
          provided.size !== current.size ||
          [...current].some((id) => !provided.has(id))
        ) {
          return {
            success: false,
            message:
              "Provide every current item ID exactly once — no missing, extra, or duplicate IDs. Call getWatchlistContents to get the current list.",
          };
        }

        const reordered = await reorderWatchlistItemsDb({
          watchlistId,
          userId,
          itemIds: trimmed,
        });
        return {
          success: reordered,
          message: reordered
            ? "Reordered watchlist items"
            : "Watchlist not found or you do not own it",
        };
      },
    }),
  };
}

export type ChatTools = ReturnType<typeof createChatTools>;
